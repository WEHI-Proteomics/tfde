import time
import zmq
import random
import pandas as pd

PUBLISHED_FRAMES_DIR = '/data/published-frames'

def consumer():
    consumer_id = random.randrange(1,10005)
    print("consumer {}".format(consumer_id))
    context = zmq.Context()
    # receive work
    consumer_receiver = context.socket(zmq.PULL)
    consumer_receiver.connect("tcp://127.0.0.1:5557")
    
    while True:
        message = consumer_receiver.recv_json()
        start_run = time.time()
        print("message received: {}".format(message))
        frame_id = message['frame_id']
        base_name = message['base_name']

        frame_file_name = '{}/{}'.format(PUBLISHED_FRAMES_DIR, base_name)
        frame_df = pd.read_pickle(frame_file_name)
        print("loaded {} points from {}".format(len(frame_df), base_name))
        stop_run = time.time()
        print("processed message in {} seconds".format(round(stop_run-start_run,1)))

consumer()
