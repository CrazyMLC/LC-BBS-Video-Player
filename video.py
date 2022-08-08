import threading,queue
import cv2
import numpy as np
import subprocess,sys,os,math
from time import sleep,perf_counter
from datetime import timedelta
if len(sys.argv) < 2:
	quit()

colorDebug = 0
# 0 does nothing, 1 outputs a video of preprocessed images, 2 pauses after every preprocessing to show you the frame that's being encoded

diff_mode = 0
# 0 for simple diff, 1 for YCrCb distance

optimize = 1
# 0 for a thorough encoding, 1 for some shortcuts, 2 for limited charset, 3 for even fewer characters and limited preprocessing, 4 is only boxes
# 1 should have no consequences over 0. 3 looks pretty good still, and encodes very quickly. 4 is very ugly but is an interesting contrast.

playback_speed = 0.35
# playback speed multiplier. while the encoder tells the player to wait X frames, the encoder will go forward X*speed frames instead.

frame_sleep = 6
# after the frame has finished loading (at 24 modified characters per second) the player will wait this many frames on the completed image, to let the viewer take it all in.

min_delay = 10
# reduces filesize and encoding time by limiting framerate in low-variance sections of the video.


def BGR2linear(nparr):# i'm not going to do the piecewise version, and colour-science is very slow
	temp = nparr.astype(np.float32)
	temp /= 255.0
	return temp ** 2.4
def linear2BGR(nparr, int_convert = True):
	temp = abs(nparr) ** (1.0/2.4)
	temp *= 255.0
	if int_convert:
		return temp.astype(np.uint8)
	else:
		return temp
def linear2GRAY(nparr, int_convert = True):
	temp = abs(nparr) ** (1.0/2.4)
	temp *= 255.0
	temp = cv2.cvtColor(abs(temp), cv2.COLOR_BGRA2GRAY)
	if int_convert:
		return temp.astype(np.uint8)
	else:
		return temp

def diff(img, color):# this is where the magic happens. if this goes wrong, the video looks like crap.
	if diff_mode == 0:
		return linear2BGR(img - color).sum(2)
	elif diff_mode == 1:# ycrcb seems a bit better at color-coding things
		new_img = cv2.cvtColor(linear2BGR(img), cv2.COLOR_BGR2YCrCb).astype(np.float32)
		new_color = cv2.cvtColor(np.array([[linear2BGR(color)]]), cv2.COLOR_BGR2YCrCb).astype(np.float32)
		return ((new_img - new_color)**2 * (2,1,1) ).sum(2)



print("Resizing and lowering fps with ffmpeg...")
subprocess.call(f'ffmpeg -loglevel error -i "{sys.argv[1]}" -an -n -vf "fps=30, scale=-1:300, pad=ceil(iw/2)*2:0" temp.mp4')
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
display_xoffset = int((504-frame_width)/18)# for dealing with videos that are too thin
pad_left = 0
pad_right = 0
if frame_width % 9 > 0:
	pad_left = 9 - (frame_width % 9)
	frame_width += pad_left
	pad_right = math.ceil(pad_left/2)
	pad_left = int(pad_left/2)
if colorDebug:
	result = cv2.VideoWriter('debug.avi',
							cv2.VideoWriter_fourcc(*'MJPG'),
							1.0, (frame_width, frame_height))


palette = BGR2linear(cv2.imread('palette.png'))[0]
# to achieve a black value of roughly 48,47,25
add = palette[0]

# to achieve a white value of roughly 244,246,172 after the add
mul = palette[-1]
mul -= add

txt_lastframe = np.full((20,int((frame_width+8)/9)),'D5')
txt_buffer = txt_lastframe.copy()

font = cv2.imread('system_bold_bw.png',0)/255.0
font_chars = 'AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz.,:;!?&#/\\%\'"0123456789+-*()[]^█▟▙▜▛▀▄▐▌▝▘▗▖─⚉═║╔╗╚╝╠╣╦╩╬>▲▼™`♦♣♠♥<☺☻ '
if optimize == 4:
	limited_set = [121]+[83]
elif optimize == 3:
	limited_set = list(range(83,96))+[51,63,121]
elif optimize == 2:
	limited_set = list(range(77,97))+[51,77,109,110,111,113,118,121]
else:
	limited_set = range(len(font_chars))
palette_grey = cv2.imread('palette.png',0)[0]
b64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'
# color (5 bits) and symbol (7 bits) perfectly fits into two base64 characters. there's even room to implement RLE.

def encode_chunk(img,diffs,bufx,bufy):
	global txt_buffer
	character = [1,1]#color,symbol
	score = float('inf')#smaller is better
	
	#let's see if we can't shortcut out of this
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(linear2GRAY(img, False))
	if max_val-(palette_grey[1]-palette_grey[0])*0.01 < palette_grey[0]:
		#if the max value of this section of the image is only just above our darkest color, this is an empty tile.
		txt_buffer[bufy][bufx] = 'D5'
		return
	if min_val+(palette_grey[-1]-palette_grey[-2])*0.01 > palette_grey[-1]:
		#if the min value of this section of the image is only just below our brightest value, this is a max brightness tile.
		txt_buffer[bufy][bufx] = 'jT'
		return
	#there's already a lot of characters to check through, the colors only make it worse.
	#so let's narrow down our options.
	palmin = 1
	palmax = 18
	if optimize > 0:
		for i in range(1,18):
			if max_val < palette_grey[i]:
				palmax = i+2
				break
		for i in range(17,0,-1):
			if min_val > palette_grey[i]:
				palmin = i-1
				break
	
	for sym in limited_set:
		char_img = font[:, sym*11 + 1:sym*11 + 10]#we're ignoring the characters that overlap with neighbors for now (% and ™)
		#let's check the black values of this character first. if the score from that alone is worse, then we can skip this character.
		#we can also reuse these black values for each of the color values
		temp_score = [np.sum(diffs[0][bufy*15:(bufy+1)*15, bufx*9:(bufx+1)*9]*(1-char_img)),0]#dark, light
		#we can use char_img as a mask to let us sum the difference values for this color. pretty sweet.
		if temp_score[0] > score:
			continue
		
		#now to check the possible colors
		if sym < 121:
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

q = queue.Queue()
def worker():
	while True:
		i = q.get()
		encode_chunk(i[0],i[1],i[2],i[3])
		q.task_done()
for i in range(32):
	threading.Thread(target=worker, daemon=True).start()

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
print("Scanning through frames...\n")
frameCount = 0
delay = 0
print(f"{round(100*playback_speed)}% playback speed.\nSleeping on rendered frames for {round(frame_sleep/30,2)} seconds.\nMinimum time between frames is {round(min_delay/30,2)} seconds.\n")
newFrame = True
maxFrames = video.get(cv2.CAP_PROP_FRAME_COUNT)
start = perf_counter()
while(newFrame):
	newFrame, frame = video.read()
	completion = int((video.get(cv2.CAP_PROP_POS_FRAMES)/maxFrames)*100)
	if newFrame:
		delay -= 1
		if delay <= 0:
			print(f"\rFrame {frameCount}: Preprocessing...   ({completion}% total, {timedelta(seconds=round(perf_counter()-start))}, {round(float(perf_counter()-start)/max(frameCount,1),1)}s per frame)", end='', flush=True)
			if crop_amount > 0:
				frame = frame[:, crop_amount:-crop_amount]
			if optimize < 3:
				frame = unsharp_mask(cv2.fastNlMeansDenoisingColored(frame,None,5,5,5,15), amount = 2.0)
			frame = (BGR2linear(frame)*mul)+add
			frame = np.pad(frame, ((0,0),(pad_left,pad_right),(0,0)))
			
			if colorDebug & 0b10:
				cv2.imshow('Frame', linear2BGR(frame))
				if cv2.waitKey(0) & 0xFF == ord('s'):
					break
			
			diffs = list(range(18))
			for d in diffs:
				diffs[d] = diff(frame, palette[d])
			
			print(f"\rFrame {frameCount}: Encoding...        ({completion}% total, {timedelta(seconds=round(perf_counter()-start))}, {round(float(perf_counter()-start)/max(frameCount,1),1)}s per frame)", end='', flush=True)
			for y in range(len(txt_buffer)):
				for x in range(len(txt_buffer[y])):
					q.put((frame[y*15:(y+1)*15, x*9:(x+1)*9],diffs,x,y))
			q.join()
			
			print(f"\rFrame {frameCount}: Compressing...     ({completion}% total, {timedelta(seconds=round(perf_counter()-start))}, {round(float(perf_counter()-start)/max(frameCount,1),1)}s per frame)", end='', flush=True)
			#print(txt_buffer)
			if frameCount > 0:
				output_txt.write(',\n"')
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
				frame = linear2BGR(frame)
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
print("\n\nThe video was successfully processed.")
print(f"{frameCount} frames were encoded in {timedelta(seconds=round(perf_counter()-start))}")
input("\nPress any key to exit.")
