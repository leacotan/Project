import Isopod_detection
import sys
import os
import time

def main(image_directory_name, dest_output_dir_name):
    for filename in os.listdir(image_directory_name):
        if int(filename) >= 1:
            if os.path.isdir(os.path.join(image_directory_name, filename)):
                Isopod_detection.main(os.path.join(image_directory_name, filename),dest_output_dir_name,190,4)

if __name__ == '__main__':
    # print("Usage: directory(path) dest_dierctory(path) light/dark(string) )")

    t0 = time.time()

    main(sys.argv[1],sys.argv[2])

    t1 = time.time()


    total = t1 - t0
    print("total time to process folder: " + str(total) + "seconds")