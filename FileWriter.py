__author__ = 'LEA'
import numpy as np

class FileWriter:
    """
    wrapper that writes to a file in a specific format
    """


    def __init__(self, directory_name,image_directory):
        """
        initializes the file writer. by wrapping a file object and wiriting the first line
        :param directory_name:
        :return:
        """
        file_num = image_directory.split("\\")[-1]
        self.file = open(directory_name + "\\"+file_num+ "_data_new_run.txt", "w")
        self.file.write("image_name"+"\t"+"num_isopodes"+"\n")



    def write_image_data(self, image_name,coordinates, num_isopods ):
        """
        writes the data taken from a single image.
        :param image_name: the name of the image file
        :param coordinates: the coordinates of the isopods
        :param num_isopods: the number of found isopods
        :return: nothing. writes to file
        """
        parsed_imname_list = image_name.split("_")
        reversed_date = parsed_imname_list[0].split("-")
        date = reversed_date[2]+"/"+reversed_date[1]+"/"+reversed_date[0]
        time = parsed_imname_list[1].replace("-",":")

        self.file.write(date+"\t"+time +"\t"+ str(num_isopods)+"\t")
        for coor in coordinates:
            self.file.write(str(int(np.floor(coor[0])))+ "\t"+str(int(np.floor(coor[1])))+"\t")
        self.file.write("\n")

    def write_background(self, scorpion_coor, soil_coor, leaf_coor,black_coor, white_coor ):
        """
        the last function called. writes the data about the background image at the end of the file.
        THE ORDER IS :
        scorpion_coo, soil_coor,leaf_coor, black_coor, white_coor
        watch out- sometimes there will be only 3 circles

        :param scorpion_coor: x,y of scorpion circle center, and r radious
        :param soil_coor: x,y of soil food circle center, and r radious
        :param leaf_coor: x,y of leaf food circle center, and r radious
        :param black_coor: x,y of black circle center, and r radious
        :param white_coor: x,y of white circle center, and r radious
        :return: nothing, wirtes to file
        """
        scor = "scorpion: " + str(scorpion_coor[0]) + "\t" + str(scorpion_coor[1])+"\t" + str(scorpion_coor[2])
        soil = "soil: " + str(soil_coor[0])+ "\t" + str(soil_coor[1])+ "\t" + str(soil_coor[2])
        leaf = "leaf: " + str(leaf_coor[0])+ "\t" + str(leaf_coor[1])+ "\t" + str(leaf_coor[2])
        black = "black: "+ str(black_coor[0])+ "\t" + str(black_coor[1])+ "\t" + str(black_coor[2])
        white = "white: "+ str(white_coor[0])+ "\t" + str(white_coor[1])+ "\t" + str(white_coor[2])
        lines = ["background_circle_centers:"+"\n" , scor+"\n",soil+"\n" ,leaf+"\n",black+"\n", white+"\n"]
        self.file.writelines(lines)
