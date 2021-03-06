import argparse
import sys
import cv2
import socket
import pickle
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from threading import Thread
from main.video_processing import VideoStream

__version__ = '0.1.2'

parser = argparse.ArgumentParser(description='video url or path')
ap = argparse.ArgumentParser()
ap.add_argument("-src", "--source", required=False, help="cam url")
ap.add_argument("-s", "--server_ip", required=False, default='0.0.0.0',
                help="ip address of the server to which the client will connect")
ap.add_argument("-p", "--port", required=False, default=14600,
                help="socket port")

args = vars(ap.parse_args())
#args["source"] = "/Users/andrey/Downloads/Telegram Desktop/vlc_record_2019_05_30_12h50m55s.mp4"
#args["source"] = "rtsp://user:Hneu74k092@10.101.106.104:554/live/main"
#args["source"] = "rtsp://admin:admin@10.101.1.221:554/ch01/1" #base stream 0
args["source"] = 0


sock = socket.socket()
sock.bind((args["server_ip"], args["port"]))
sock.listen(10)
sock.setblocking(0)


class CamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        print(self.path)
        if self.path.endswith('/stream.mjpg'):
            self.send_response(20)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
            self.end_headers()
            while True:
                try:
                    r, buf = cv2.imencode(".jpg", frame)
                    self.wfile.write("--jpgboundary\r\n".encode())
                    self.end_headers()
                    self.wfile.write(bytearray(buf))
                except KeyboardInterrupt:
                    break
            return

        if self.path.endswith('.html') or self.path == "/":
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write('<html><head></head><body>')
            self.wfile.write('<img src="http://localhost:9090/stream.mjpg" height="240px" width="320px"/>')
            self.wfile.write('</body></html>')
            return


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""


class WebcamVideoStream:
    def __init__(self, src=0):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        # self.stream.set(3, 1920)
        # self.stream.set(4, 1080)
        # self.stream.set(15,-100)
        (self.grabbed, self.frame) = self.stream.read()

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                self.stream.release()
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


def main(args=None):
    global frame
    ip = args["server_ip"]
    try:
        if args["source"] == '0':
            camera_url = int(args["source"])
        else:
            camera_url = args["source"]

        # connection = EventDBLogger()
        # table = connection.create_table(camera_url)

        vs = VideoStream(camera_url)
        server = ThreadedHTTPServer((ip, 9091), CamHandler)
        print("[INFO] starting server")
        target = Thread(target=server.serve_forever,args=())

        i = 0
        while True:
            frame, _, _ = vs.process_next_frame()
            if(i == 0):
                target.start()
            i +=1
            try:
                client, addr = sock.accept()
                print("accept")
            except socket.error:  # данных нет
                pass  # тут ставим код выхода
            else:  # данные есть
                client.setblocking(0)  # снимаем блокировку и тут тоже
                query = client.recv(16384)
                query = pickle.loads(query)
                print("Request type: " + query["type"])
                if query["type"] == "get_objects":
                    message = {}
                    for i in vs.trackableObjects.keys():
                        tmp = {
                            #'objectID': vs.trackableObjects[i].objectID,
                            'name':  vs.trackableObjects[i].name,
                            'names': vs.trackableObjects[i].names
                        }
                        message[i] = tmp
                    b_message = pickle.dumps(message)
                    client.send(b_message)
                    client.close()
                    print("close")
                elif query["type"] == "set_name":
                    object_id = int(query['object_id'])
                    vs.trackableObjects[object_id].name = (query['name'], )
                    #vs.trackableObjects[object_id].stars = query['stars']
                    #vs.trackableObjects[object_id].description = query['description']
                    print("close")


    except KeyboardInterrupt:
        sys.exit()


if __name__ == '__main__':
    main(args)
