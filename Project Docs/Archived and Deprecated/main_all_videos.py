from utils import *
import pandas as pd


# initialize the working directory
initialize()


list_of_video = []
for dirpath, dirnames, filenames in os.walk('Source'):
    list_of_video = filenames

print(f'There are {len(list_of_video)} videos.')


# convert video to images
# (you may only do the video processing once in order to save time)
video_processing(list_of_video)

df = pd.DataFrame(np.array(list_of_video), columns=['video'])

index1_list = []
index2_list = []
index3_list = []
index4_list = []
index5_list = []
index6_list = []
index7_list = []
index8_list = []
index9_list = []
index10_list = []


video_index = 1
for video in list_of_video:

    print(f"{video_index}/{len(list_of_video)}: {video}")

    apex_x, apex_y = apex_evolution(video)
    index1_list.append(max(apex_y)-min(apex_y))

    peak_x, peak_y = peak_evolution(video)
    x_left = min(peak_x[:, 0])
    x_right = max(peak_x[:, 1]) + peak_x[0, 0]
    index2_list.append(x_right - x_left)

    image_path = f'Data/video{video}/image1.jpg'
    image = cv2.imread(image_path)
    image = image_processing(image)
    thickness0 = np.array(get_thickness(image, x_left=0, x_right=576, output=True))
    index3_list.append(np.mean(thickness0))

    thickness, x_min = thickness_evolution(video, x_left=0, x_right= 576, p_left=x_left, p_right=x_right)
    index4_list.append(np.mean(thickness))

    index5_list.append(x_min-apex_x)

    index6_list.append(np.mean(thickness[:, int((x_min-x_left)/10):int((x_min-x_left)/10)+1]))

    critical_pressure = displacement_evolution(video)
    index7_list.append(critical_pressure/(max(apex_y)-min(apex_y)))

    index8_list.append(get_viscosity(video))

    index9_list.append(get_parabola_form(video, output=True))

    video_index += 1
    
      
df['index1(maximum applanation of apex)'] = np.array(index1_list)
df['index2(maximum horizontal distance between two peaks)'] = np.array(index2_list)
df['index3(initial average thickness)'] = np.array(index3_list)
df['index4(average thickness change between two peaks)'] = np.array(index4_list)
df['index5(horizontal distance between initial thickness minimum and apex)'] = np.array(index5_list)
df['index6(average thickness change around initial thickness minimum)'] = np.array(index6_list)
df['index7(effective stiffness)'] = np.array(index7_list)
df['index8(effective viscosity)'] = np.array(index8_list)
df['index9(parabola form)'] = np.array(index9_list)


df.to_csv('Results/results.csv', index=False)
