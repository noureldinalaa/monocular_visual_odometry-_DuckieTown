import csv
import numpy as np
import pickle

import matplotlib.pyplot as plt
import PIL
from PIL import Image

class Rotation_Translation(object):
    # def __init__(self,input_points,output_points):
    #     self.input_points =input_points
    #     self.output_points = output_points

    def read_input_points(self):
        with open('inputs.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:

                if line_count == 0:
                    input_points = np.array([[row[0], row[1]]])
                    output_points = np.array([[row[2], row[3]]])
                    line_count += 1
                    print(f'\t input ----> this is point_0 {input_points[0][0]} this is point_1 {input_points[0][1]}.')
                    print(f'\t output --> this is point_0 {output_points[0][0]} this is point_1 {output_points[0][1]}.')
                else:
                    input_flag = np.array([[row[0], row[1]]])
                    input_points = np.append(input_points, input_flag, axis=0)
                    output_flag = np.array([[row[2], row[3]]])
                    output_points = np.append(output_points, output_flag, axis=0)
                    # input_points.append([row[0],row[1]],axis = 1)
                    print(input_points.shape)
                    print(
                        f'\t input ----> this is point_0 {input_points[line_count][0]} this is point_1 {input_points[line_count][1]}.')
                    print(
                        f'\t output --> this is point_0 {output_points[line_count][0]} this is point_1 {output_points[line_count][1]}.')
                    line_count += 1

            #         input_flag = [[row[0],row[1]]]
            #         input_points.append(input_flag)
            #         print(input_points.shape)
            #         print(f'\t input ----> this is point_1 {input_points[0][0]} this is point_2 {input_points[0][1]}.')
            #         output_flag = [[row[2],row[3]]]
            #         output_points.append(output_flag)
            #         print(f'\t output ---> this is point_1 {output_points[0][0]} this is point_2 {output_points[0][1]}.')
            #
            print(f'Processed {line_count} lines.')

        saveObject = (input_points,output_points)
        with open("./input_output_points.pickle","wb") as file :
            pickle.dump(saveObject,file)

        return input_points,output_points

    def input_points_length(self):
        with open('inputs_length.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    input_length = np.array([row[0]])
                    line_count += 1
                    # print(f'\t input length = {input_length[0]} ')
                else:
                    input_length_flag = np.array([row[0]])
                    input_length = np.append(input_length, input_length_flag, axis=0)
                    line_count += 1
                    # print(f'\t input length ={input_length} ')

        max_number = max(input_length)
        # print(max_number)
        # print(input_length)

        point_numbers = len(input_length)
        # print(point_numbers)

        return input_length , max_number,point_numbers

    def env_data_points(self):
        with open('dataset.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=' ')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    env_points = np.array([[row[0], row[1]]])
                    env_angles = np.array([[row[2]]])
                    line_count += 1
                else:
                    env_points_flag = np.array([[row[0], row[1]]])
                    env_points = np.append(env_points, env_points_flag, axis=0)
                    env_angles_flag = np.array([[row[2]]])
                    env_angles = np.append(env_angles, env_angles_flag, axis=0)
                    line_count += 1


        # print(env_points)
        env_length = len(env_points)
        # print(env_length)

        return env_points ,env_angles,env_length


    def env_distance_between_points(self,env_points ,env_length,point_numbers,input_length):
        index = 0
        for i in range(env_length - 1):
            if i == 0:
                #         x_env = np.array([float(env_points[index +1][0]) -float(env_points[index][0])])
                #         y_env = np.array([float(env_points[index +1][1]) -float(env_points[index][1])])
                dist = np.array([np.sqrt((((float(env_points[index + 1][0]) - float(env_points[index][0])) ** 2) +
                                          ((float(env_points[index + 1][1]) - float(env_points[index][1])) ** 2)))])
                # print(dist)
                index += 1

            else:
                #         x_env_flag = np.array([float(env_points[index +1][0]) -float(env_points[index][0])])
                #         x_env= np.append(x_env,x_env_flag,axis =0)
                #         y_env_flag = np.array([float(env_points[index +1][1]) -float(env_points[index][1])])
                #         y_env= np.append(x_env,x_env_flag,axis =0)
                dist_flag = np.array([np.sqrt((((float(env_points[index + 1][0]) - float(env_points[index][0])) ** 2) +
                                               ((float(env_points[index + 1][1]) - float(
                                                   env_points[index][1])) ** 2)))])
                dist = np.append(dist, dist_flag, axis=0)
                index += 1
        # print(dist.shape)
        # print("this is distance-------------------->" + str(dist))
        #this to make each line in the optical flow coressponding it the distance (if it is repeated 50 points in one frame
        # then there will be 50 same dist)
        distance = []
        count1 = 0
        count2 = 0
        count_number = []
        for i in range(point_numbers):
            points_array = input_length[i]
            dist_flag = dist[i]
            for j in range(int(points_array)):
                distance.append(dist_flag)
            count1 = len(distance)
            count = count1 - count2
            # print(count)
            count_number.append(count)
            count2 = count1

        # print(len(distance))

        return distance

    def env_angles_between_points(self,env_angles, env_length,point_numbers,input_length):
        index = 0
        for i in range(env_length - 1):
            if i == 0:
                env_rotation = np.array([float(env_angles[index + 1][0]) - float(env_angles[index][0])])
                index += 1

            else:
                env_rotation_flag = np.array([float(env_angles[index + 1][0]) - float(env_angles[index][0])])
                env_rotation = np.append(env_rotation, env_rotation_flag, axis=0)
                index += 1
        # print(env_rotation.shape)
        # print("this is rotation-------------------->" + str(env_rotation))
        #this to make each line in the optical flow coressponding it the rotation(if it is repeated 50 points in one frame
        # then there will be 50 same rotation)
        rotation = []
        count1 = 0
        count2 = 0
        count_number = []
        for i in range(point_numbers):
            points_array = input_length[i]
            rotation_flag = env_rotation[i]
            for j in range(int(points_array)):
                rotation.append(rotation_flag)
            count1 = len(rotation)
            count = count1 - count2
            # print(count)
            count_number.append(count)
            count2 = count1

        # print(len(rotation))
        return rotation

    def optical_points_distance_histogram(self,x_difference,y_difference,input_length):

        power_x_difference = np.power(x_difference, 2)
        power_y_difference = np.power(y_difference, 2)
        add_x_y_difference = np.add(power_x_difference, power_y_difference)
        distance_in_frames = np.sqrt(add_x_y_difference)
        # print(distance_in_frames[-1])
        # print(len(distance_in_frames))
        length1 = int(np.round(len(distance_in_frames) * 0.8))
        # print(length1)
        length2 = int(np.round(799 * 0.8))
        # print(length2)
        loop_var = 0
        loop_var2 = 0
        dist_histogram = []
        for i in range(length2):
            loop_var = int(input_length[i])
            for j in range(loop_var):
                dist_histogram.append(distance_in_frames[loop_var2 + j])

            loop_var2 += int(input_length[i])
            plt.hist(dist_histogram, bins=10)
            # plt.title("ground truth")
            plt.xlabel("std = " + str(np.std(dist_histogram)))
            # plt.show(block = False)
            # plt.pause(1.5)
            plt.savefig('./plot/' + 'plot_' + str(i) + '.png')
            plt.close('all')
            # print("std = " + str(np.std(dist_histogram)))

            dist_histogram = []

            im1 = Image.open('mixed_plots/self.id' + str(i) + '.png')
            # im2 = Image.open('undistorted_plots/self.id' + str(i) + '_1.png')
            im2 = Image.open('plot/plot_' + str(i) + '.png')

            imgs = [im1, im2]
            # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
            min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
            imgs_comb = np.hstack((np.asarray(i.resize(min_shape)) for i in imgs))

            # save that beautiful picture
            imgs_comb = PIL.Image.fromarray(imgs_comb)
            imgs_comb.save('./image/comb_img_' + str(i) + '.png')

    def env_dist_rotation_histogram(self,translation_rotation2,input_length):
        length3 = int(np.round(799 * 0.8))
        distance_trained = []
        rotation_trained = []
        skip_var = 0
        tran_rot = translation_rotation2.cpu().detach().numpy()

        for i in range(int(length3)):
            distance_trained.append(tran_rot[skip_var][0])
            rotation_trained.append(tran_rot[skip_var][1])
            skip_var += int(input_length[i])
            print(skip_var)

        plt.hist(distance_trained, bins=10)
        plt.title("env_dist_trained")
        plt.xlabel("std = " + str(np.std(distance_trained)))
        plt.savefig('./trained_distance_histograms/' + 'distance_trained.png')
        plt.show()

        plt.hist(rotation_trained, bins=10)
        plt.title("rotation_trained")
        plt.xlabel("std = " + str(np.std(rotation_trained)))
        plt.savefig('./trained_distance_histograms/' + 'rotation_trained.png')
        plt.show()



