import threading
import cv2
import numpy as np
import subprocess,sys,os
from time import sleep,perf_counter
colorDebug = 0

if len(sys.argv) < 2:
	quit()

#i'm not going to do the piecewise version, and colour-science is very slow
def sBGRA2linear(nparr):
	temp = nparr.astype(np.float32)
	temp /= 255.0
	return temp ** 2.4
def linear2sBGRA(nparr):
	temp = abs(nparr) ** (1.0/2.4)
	temp *= 255.0
	return temp.astype(np.uint8)

print("Resizing and lowering fps with ffmpeg...")
subprocess.call(f'ffmpeg -i "{sys.argv[1]}" -an -n -hide_banner -loglevel error -vf "fps=30, scale=-1:300, pad=ceil(iw/2)*2:0" temp.mp4')
print("Capturing ffmpeg output...")
video = cv2.VideoCapture('temp.mp4')
if (video.isOpened() == False): 
	print("Error reading video file")
	quit()
frame_height = int(video.get(4))# should always be 300 but it's good to check
frame_width = int(video.get(3))
crop_amount = int(max(0,frame_width - 504)/2)# the terminal is only 504 pixels wide
if crop_amount:
	frame_width = int(frame_width - 2*crop_amount)
display_xoffset = (504-frame_width)/2# for dealing with videos that are too thin

if colorDebug:
	result = cv2.VideoWriter('debug.avi',
							cv2.VideoWriter_fourcc(*'MJPG'),
							1.0, (frame_width, frame_height))


# to achieve a black value of roughly 48,47,25
add = sBGRA2linear(np.array((48, 47, 25)))

# to achieve a white value of roughly 244,246,172 after the add
mul = sBGRA2linear(np.array((244, 246, 172)))
mul -= add

txt_lastframe = np.full((20,int((frame_width+8)/9)),'D5')
txt_buffer = txt_lastframe.copy()


font = cv2.imread('system_bold_bw.png',0)/255.0
#chars = 'AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz.,:;!?&#/\\%\'"0123456789+-*()[]^█▟▙▜▛▀▄▐▌▝▘▗▖─⚉═║╔╗╚╝╠╣╦╩╬>▲▼™`♦♣♠♥<☺☻ '
palette = sBGRA2linear(cv2.imread('palette.png'))[0]
palette_grey = cv2.cvtColor(sBGRA2linear(cv2.imread('palette.png')), cv2.COLOR_BGRA2GRAY)[0]
b64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'
# color (5 bits) and symbol (7 bits) perfectly fits into two base64 characters. there's even room to implement RLE.

def encode_chunk(img,diffs,bufx,bufy):
	global txt_buffer
	character = [1,1]#color,symbol
	score = float('inf')#smaller is better
	
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY))
	if max_val-0.001 < palette_grey[0]:
		#print("black")
		txt_buffer[bufy][bufx] = 'D5'
		return
	if min_val+0.1 > palette_grey[-1]:
		#print("white")
		txt_buffer[bufy][bufx] = 'jT'
		return
	#there's already a lot of characters to check through, the colors only make it worse.
	#so let's narrow down our options.
	palmin = 1
	palmax = 18
	for i in range(1,18):
		if max_val < palette_grey[i]:
			palmax = i+2
			break
	for i in range(17,0,-1):
		if min_val > palette_grey[i]:
			palmin = i-1
			break
	
	for sym in range(int(len(font[0])/11)):
		char_img = sym*11 + 1
		char_img = font[:, char_img:char_img+9]#we're ignoring the characters that overlap with neighbors for now (% and ™)
		#let's check the black values of this character first. if the score from that alone is worse, then we can skip this character.
		#we can also reuse these black values for each of the color values
		temp_score = [0,0]#dark, light
		temp_score[0] = np.sum(diffs[0][bufy*15:(bufy+1)*15, bufx*9:(bufx+1)*9]*(1-char_img))
		#we can use char_img as a mask to lot us sum the difference values for this color. pretty sweet.
		if temp_score[0] > score:
			continue
		
		#now to check the possible colors
		for col in range(max(1,palmin),min(18,palmax)):
			temp_score[1] = np.sum(diffs[col][bufy*15:(bufy+1)*15, bufx*9:(bufx+1)*9]*char_img)
			if sum(temp_score) < score:
				character = [col, sym]
				score = sum(temp_score)
		#we should have found the best character by now
	#now to encode into base 64
	result = f'{b64[(character[0] << 1) + (character[1] & 0b1000000 > 0)]}{b64[character[1] & 0b111111]}'
	txt_buffer[bufy][bufx] = result
	#print(character,score,temp_score)

def encode_rle(txt):
	flat = txt.flatten()
	result = flat[0]
	i = 1
	while i < len(flat):
		if i+1 < len(flat) and flat[i-1] == flat[i] == flat[i+1]:
			result += b64[0]
			count = 2
			while i+count < len(flat) and flat[i+count] == flat[i+count-1] and count-1 < len(b64):
				count += 1
			i += count
			result += b64[count-2]
		else:
			result += flat[i]
			i+=1
	return result
		
		
def end():
	video.release()
	cv2.destroyAllWindows()
	os.remove('temp.mp4')
	if colorDebug:
		result.release()
		#os.system('debug.avi')


#taken from https://stackoverflow.com/questions/4993082/how-can-i-sharpen-an-image-in-opencv
def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
	"""Return a sharpened version of the image, using an unsharp mask."""
	blurred = cv2.GaussianBlur(image, kernel_size, sigma)
	sharpened = float(amount + 1) * image - float(amount) * blurred
	sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
	sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
	sharpened = sharpened.round().astype(np.uint8)
	if threshold > 0:
		low_contrast_mask = np.absolute(image - blurred) < threshold
		np.copyto(sharpened, image, where=low_contrast_mask)
	return sharpened

output_txt = open(os.path.basename(sys.argv[1])+'.js','w', encoding="utf-8")
output_txt.write('let frames = ["')
print("Scanning through frames...")
frameCount = 0
delay = 0


playback_speed = 0.35
frame_sleep = 6
min_delay = 10
print(f"{round(100*playback_speed)}% playback speed.\nSleeping on rendered frames for {round(frame_sleep/30,2)} seconds.\nMinimum time between frames is {round(min_delay/30,2)} seconds.")
newFrame = True
start = perf_counter()
while(newFrame):
	newFrame, frame = video.read()
	completion = int(video.get(cv2.CAP_PROP_POS_AVI_RATIO)*100)
	if newFrame:
		delay -= 1
		if delay <= 0:
			diffs = list(range(18))
			print(f"\rFrame {frameCount}: Preprocessing... ({completion}%)   ", end='', flush=True)
			frame = unsharp_mask(cv2.fastNlMeansDenoisingColored(frame,None,5,5,5,15), amount = 2.0)
			frame = (sBGRA2linear(frame[:, crop_amount:-crop_amount])*mul)+add
			
			if colorDebug & 0b10:
				cv2.imshow('Frame', linear2sBGRA(frame))
				if cv2.waitKey(0) & 0xFF == ord('s'):
					break
			
			img_buff = frame.copy()
			for d in diffs:
				#print("calc diff",d)
				#print("init",img_buff[:2,:2])
				img_buff = linear2sBGRA(frame - palette[d])
				#print("sub",img_buff[:2,:2])
				#img_buff = abs(img_buff)
				#print("sqr",img_buff[:2,:2])
				diffs[d] = img_buff.sum(2)
			
			threads = []
			print(f"\rFrame {frameCount}: Encoding...      ({completion}%)   ", end='', flush=True)
			for y in range(len(txt_buffer)):
				for x in range(len(txt_buffer[y])):
					threads.append(threading.Thread(target=encode_chunk, args=(frame[y*15:(y+1)*15, x*9:(x+1)*9],diffs,x,y)))#[y*15:(y+1)*15, x*9:(x+1)*9]
					threads[-1].start()
			for thread in threads:
				thread.join()
			print(f"\rFrame {frameCount}: Compressing...   ({completion}%)   ", end='', flush=True)
			#print(txt_buffer)
			if frameCount > 0:
				output_txt.write(',\n"')
				output_txt.flush()
			output_txt.write(encode_rle(txt_buffer))
			# frames to skip: playback_speed*max(updated_tiles/24 + frame_sleep,min_delay) / (30/fps)
			#print("\n",np.sum(txt_buffer != txt_lastframe))
			delay = min(max(np.sum(txt_buffer != txt_lastframe)/24.0 + frame_sleep, min_delay),65)
			output_txt.write(b64[int(delay)-2]+'"')
			delay *= playback_speed
			txt_lastframe = txt_buffer.copy()
			output_txt.flush()
			frameCount += 1
			if colorDebug:
				frame = linear2sBGRA(frame)
		if colorDebug:
				result.write(frame)
output_txt.write(f"""];
let width = {len(txt_buffer[0])};
let offset = {int(display_xoffset)};
let name = '{os.path.basename(sys.argv[1])}';
let font = 'AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz.,:;!?&#/\\\\%\\'"0123456789+-*()[]^█▟▙▜▛▀▄▐▌▝▘▗▖─⚉═║╔╗╚╝╠╣╦╩╬>▲▼™`♦♣♠♥<☺☻ ';
let b64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';

var curFrame;
var displayed;
var counter;
var playing;
var showUI;

function getName() {{
    return name;
}}

function onConnect() {{
    curFrame = 0;
	counter = 1;
	displayed = 0;
	playing = true;
	showUI = true;
}}

function onUpdate() {{	
	if (playing) {{
		curFrame = curFrame % frames.length;
		if (curFrame < 0) {{
			curFrame += frames.length;
		}}
		counter--;
		if (counter == 0) {{
			curFrame++;
			if (curFrame >= frames.length) {{
				playing = false;
			}}
		}}
	}}
	if (displayed != curFrame) {{
		render(true);
		displayed = curFrame;
	}}
	if (showUI) {{
		drawText(curFrame.toString()+"/"+frames.length.toString(),17-7*playing,0,0);
		
		drawText(name,17-7*playing,0,19);
	}}
}}

function onInput(key) {{
	switch(key) {{
		case 32:
			playing = !playing;
			break;
		case 19:
			curFrame--;
			break;
		case 20:
			curFrame++;
			break;
		default:
			showUI = !showUI;
			if (!showUI) {{
				render(false);
			}}
			break;
	}}
}}

function render(set_count) {{
	clearScreen();
	if (frames[curFrame] == undefined) {{
		return;
	}}
	var frame = frames[curFrame];
	var last = [0,"A"];
	var readpos = 0;
	for (var i = 0; i < width*20; readpos++) {{
		var num1 = frame[readpos*2];
		var num2 = frame[readpos*2+1];
		num1 = b64.indexOf(num1);
		num2 = b64.indexOf(num2);
		if (num1 == 0) {{
			num2+=2;
			while (num2 > 0) {{
				drawText(last[1],last[0],(i%width)+offset,i/width|0);
				i++;
				num2--;
			}}
		}} else {{
			var col = num1 >> 1;
			var sym = font[ ((num1 & 1) << 6) + num2 ];
			if (col == undefined || sym == undefined) {{
				drawText("error",17,(i%width)+offset,i/width|0);
				return
			}}
			drawText(sym,col,(i%width)+offset,i/width|0);
			last = [col,sym]
			i++;
		}}
	}}
	if (set_count) {{
		counter = b64.indexOf(frames[curFrame].slice(-1))+2;
	}}
}}
""")
output_txt.close()
end()
print("The video was successfully processed.")
print(f"{frameCount} frames were encoded in {round(perf_counter()-start,1)} seconds.")
input("\nPress any key to exit.")
