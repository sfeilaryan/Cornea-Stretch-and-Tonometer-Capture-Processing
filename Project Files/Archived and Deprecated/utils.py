import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import sympy as sp


# define global variables

width = 576
height = 200
time = 139

mode = 6


# add necessary directories to the root path
def initialize():
    if not os.path.exists("Source/"):
        os.mkdir("Source/")
    if not os.path.exists("Data/"):
        os.mkdir("Data/")
    if not os.path.exists("Results/"):
        os.mkdir("Results/")


# visualize one instant of the video
def cv_show(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# visualize one instant of the video
def plt_show(name, image):
    plt.figure()
    plt.title(name)
    plt.imshow(image, "gray")
    plt.show()


def video_processing(list_of_video):
    for video in list_of_video:
        video_to_image(video)


def video_to_image(video):
    """
    This function transforms a video into separate images and saves them at the directory Data.
    :param video: the path of the video
    :return: no return
    """

    input_path = "Source/" + video
    if os.path.exists(input_path):
        print(f"Video of path {input_path} exists, loading video.")
    else:
        print(f"Video of path {input_path} does not exist, please check input!")
        exit(0)

    vid = cv2.VideoCapture(input_path)

    output_path = "Data/video" + str(video)
    if not os.path.exists(output_path):
        os.makedirs("Data/video" + str(video))

    img = 1
    while True:
        ret, frame = vid.read()
        if ret:
            cv2.imwrite("Data/video" + str(video) + "/image" + str(img) + ".jpg", frame)
            img += 1
        else:
            print(
                f"Video {video} converts over, {img-1} images are extracted in total."
            )
            break


def image_processing(image):
    """
    This function do image processing which is composed by many step.
    Firstly, transform the image from BGR to GRAY.
    Secondly, set useless points to zero.
    Finally, do the dilatation of some points.
    :param image: the image takes from the directory Data/
    :return: the processed image in the format of a numpy array
    """

    # converts the image from the format BGR to GRAY
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # converts the image from GRAY to black/white
    # If the intense of the pixel is bigger than a thresh, it will be regarded as a white pixel。
    # Inversely, if the intense of a pixel is smaller than this thresh, it will be regarded as a black pixel.
    # In this program, the thresh is 24 for all images, which is found based on the first image of a certain video.
    # TODO: is this thresh suitable for all images?
    image = cv2.threshold(image, 28, 255, cv2.THRESH_BINARY)[1]
    image[180:, 520:] = 0
    image[160:, 35:180] = 0
    for i in range(35):
        image[185 - int(0.68 * np.floor(i)) :, i] = 0

    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            if not (0 in [image[i + 1, j], image[i, j - 1], image[i, j + 1]]):
                image[i, j] = 255
            if not (0 in [image[i - 1, j], image[i, j - 1], image[i, j + 1]]):
                image[i, j] = 255
            if not (0 in [image[i - 1, j], image[i + 1, j], image[i, j + 1]]):
                image[i, j] = 255
            if not (0 in [image[i - 1, j], image[i + 1, j], image[i, j - 1]]):
                image[i, j] = 255

    return image


def pressure_time_curve_up(output=False):

    pressure_list = []
    for t in range(10, 18):
        pressure_list.append(0.69 * (t - 10) + 0.09)
    for t in range(18, 61):
        pressure_list.append(2.39 * (t - 18) + 5.8)

    if output:
        plt.figure()
        plt.title("Temporal distribution of the airflow pressure")
        plt.xlabel(r"time($\times$0.23ms)")
        plt.ylabel("pressure (mmHg)")
        plt.plot(np.linspace(10, 60, 51), pressure_list)
        plt.show()

    return np.array(pressure_list)


def pressure_time_curve_down(output=False):

    pressure_list = []
    for t in range(69, 100):
        pressure_list.append(-3.01 * (t - 69) + 98.89)
    for t in range(100, 109):
        pressure_list.append(-0.51875 * (t - 100) + 4.37)

    if output:
        plt.figure()
        plt.title("Temporal distribution of the airflow pressure")
        plt.xlabel(r"time($\times$0.23ms)")
        plt.ylabel("pressure (mmHg)")
        plt.plot(np.linspace(69, 108, 40), pressure_list)
        plt.show()

    return np.array(pressure_list)


def get_apex(image):
    """
    This function takes a processed image as input and returns the coordinates of its apex.
    :param image: at processed image.
    :return: the coordinates of the image in a format of integer.
    """

    # initialization: if the apex is not found, the program will end by raising an error.
    x = -1
    y = -1

    # loop to find the coordinate y of the apex: search the line one by one until there is a white pixel found in it;
    # list all white pixels in this line; take the average of this list as the apex.
    for i in range(image.shape[0] - 15):
        image_line = image[i]
        if 255 in image_line:
            y = i
            x_list = []
            for j in range(image.shape[1]):
                if not (0 in image[i : i + 15, j]):
                    x_list.append(j)
            if len(x_list) > 0:
                x = x_list[int(len(x_list) / 2)]
                break

    return int(x), int(y)


def get_apex_y(image, x):
    image_column = image[:, x]
    for i in range(image.shape[0] - 5):
        if not (0 in image_column[i : i + 5]):
            return i
    return -10


def apex_evolution(video, output=True):

    apex_x = 0
    apex_y = np.zeros(time, dtype=np.int16)

    for i in range(time):
        image_path = f"Data/video{video}/image{i+1}.jpg"
        image = cv2.imread(image_path)
        image = image_processing(image)
        if i == 0:
            apex_x, apex_y[i] = get_apex(image)
            print(f"The initial apex of video {video} is ({apex_x},{apex_y[i]}).")
            plt.figure()
            plt.imshow(image, "gray")
            plt.plot(np.arange(width), apex_y[i] * np.ones(width), "r")
            plt.plot(apex_x * np.ones(height), np.arange(height), "r")
            # plt.show()
        else:
            apex_y[i] = get_apex_y(image, apex_x)

    if not os.path.exists("Results/apex_evolution"):
        os.mkdir("Results/apex_evolution")

    if output:
        plt.figure()
        plt.title(f"Apex evolution of video {video}")
        plt.xlabel("time")
        plt.ylabel("coordinate y")
        # plt.ylim((20, 150))
        plt.gca().invert_yaxis()
        plt.plot(np.arange(time), apex_y, "+")
        plt.savefig(f"Results/apex_evolution/{video}.png")
        # plt.show()

    print(f"-- apex evolution of video {video} completed")

    return apex_x, apex_y


def peak_evolution(video, output=True):

    peak_x = np.zeros((time, 2), dtype=np.int16)
    peak_y = np.zeros((time, 2), dtype=np.int16)

    for t in range(time):
        image_path = f"Data/video{video}/image{t+1}.jpg"
        image = cv2.imread(image_path)
        image = image_processing(image)
        if t == 0:
            [peak_x[t, 0], peak_y[t, 0]] = get_apex(image)
            [peak_x[t, 1], peak_y[t, 1]] = [0, peak_y[t, 0]]
            # print(f'The initial peak for video {video} is ({peak_x[t, 0]}, {peak_y[t, 0]}).')
        else:
            [peak_x[t, 0], peak_y[t, 0]] = get_apex(image[:, : peak_x[0, 0]])
            [peak_x[t, 1], peak_y[t, 1]] = get_apex(image[:, peak_x[0, 0] :])

    if not os.path.exists("Results/peak_evolution"):
        os.mkdir("Results/peak_evolution")

    if output:

        plt.figure()
        plt.title(f"Peak evolution of video {video}")
        plt.xlabel("time")
        plt.ylabel("coordinate x")
        # plt.ylim((50, 500))
        plt.gca().invert_yaxis()
        plt.plot(np.arange(time), peak_x[:, 0], "+", label="left peak")
        plt.plot(np.arange(time), peak_x[:, 1] + peak_x[0, 0], "+", label="right peak")
        plt.legend()
        plt.savefig(f"Results/peak_evolution/{video}_x.png")
        # plt.show()

        plt.figure()
        plt.title(f"Peak evolution of video {video}")
        plt.xlabel("time")
        plt.ylabel("coordinate y")
        # plt.ylim((20, 110))
        plt.gca().invert_yaxis()
        plt.plot(np.arange(time), peak_y[:, 0], "+", label="left peak")
        plt.plot(np.arange(time), peak_y[:, 1], "+", label="right peak")
        plt.legend()
        plt.savefig(f"Results/peak_evolution/{video}_y.png")
        # plt.show()

    print(f"-- peak evolution of video {video} completed")

    return peak_x, peak_y


def displacement_evolution(video, output=True):

    t1 = 0

    displacement_up = np.zeros((51, width), dtype=np.int16)
    for t in range(10, 61):
        image_path = f"Data/video{video}/image{t+1}.jpg"
        image = cv2.imread(image_path)
        image = image_processing(image)
        for x in range(width):
            if t == 10:
                displacement_up[0, x] = get_apex_y(image, x)
            else:
                displacement_up[t - 10, x] = (
                    get_apex_y(image, x) - displacement_up[0, x]
                )
                # note the first instant that the displacement is bigger than 25 as t1
                if t1 == 0 and 500 > x > 50 and displacement_up[t - 10, x] > 30:
                    t1 = t
    displacement_up[0] = np.zeros(width)

    displacement_down = np.zeros((40, width), dtype=np.int16)
    for t in range(109, 69, -1):
        image_path = f"Data/video{video}/image{t+1}.jpg"
        image = cv2.imread(image_path)
        image = image_processing(image)
        for x in range(width):
            if t == 109:
                displacement_down[39, x] = get_apex_y(image, x)
            else:
                displacement_down[t - 70, x] = (
                    get_apex_y(image, x) - displacement_down[39, x]
                )
    displacement_down[39] = np.zeros(width)

    if not os.path.exists("Results/displacement_evolution"):
        os.mkdir("Results/displacement_evolution")

    if output:
        x_up_grid, y_up_grid = np.meshgrid(pressure_time_curve_up(), np.arange(width))
        x_down_grid, y_down_grid = np.meshgrid(
            pressure_time_curve_down(), np.arange(width)
        )
        fig = plt.figure(dpi=200)
        fig.suptitle(f"Displacement evolution of video {video}")
        fig.text(
            0.03, 0.5, "coordinate x", va="center", rotation="vertical", fontsize=12
        )
        fig.text(0.5, 0.03, "pressure", va="center", ha="center", fontsize=12)
        ax = fig.add_subplot(121)
        bx = fig.add_subplot(122)
        bx.invert_xaxis()
        cm = plt.cm.get_cmap("jet")
        ax.pcolormesh(
            x_up_grid, y_up_grid, displacement_up.T, cmap=cm, vmin=-100, vmax=100
        )
        b = bx.pcolormesh(
            x_down_grid, y_down_grid, displacement_down.T, cmap=cm, vmin=-100, vmax=100
        )
        fig.colorbar(b, ax=[ax, bx])
        plt.savefig(f"Results/displacement_evolution/{video}.png")
        # plt.show()

    print(f"-- displacement evolution of video {video} completed")

    # return the pressure corresponded to t1
    return pressure_time_curve_up()[t1 - 10]


def get_distance(image):
    distance = np.zeros(image.shape[1], dtype=np.int16)
    for j in range(image.shape[1]):
        image_column = image[:, j]
        y_min = 0
        y_max = 0
        for i in range(image.shape[0]):
            if not (0 in image_column[i : i + 10]):
                y_min = i
                break
        for k in range(y_min, image.shape[0]):
            if image_column[k] == 255:
                y_max = k
            else:
                break
        distance[j] = y_max - y_min
    return distance


def distance_evolution(video, output=True):

    distance = np.zeros((time, width), dtype=np.int16)

    for t in range(time):
        image_path = f"Data/video{video}/image{t+1}.jpg"
        image = cv2.imread(image_path)
        image = image_processing(image)
        if t == 0:
            distance[t] = get_distance(image)
        else:
            distance[t] = get_distance(image) - distance[0]
    distance[0] = np.zeros(width)

    if not os.path.exists("Results/distance_evolution"):
        os.mkdir("Results/distance_evolution")

    if output:
        x = np.arange(time)
        y = np.arange(width)
        x_grid, y_grid = np.meshgrid(x, y)
        plt.figure()
        plt.title(f"Distance evolution of video {video}")
        plt.xlabel("time")
        plt.ylabel("coordinate x")
        plt.gca().invert_yaxis()
        cm = plt.cm.get_cmap("jet")
        plt.pcolormesh(x_grid, y_grid, distance.T, cmap=cm, vmin=-10, vmax=10)
        plt.colorbar()
        plt.savefig(f"Results/distance_evolution/{video}.png")
        # plt.show()

    print(f"-- distance evolution of video {video} completed")

    return distance


def distance_visualization(video):

    if not os.path.exists("Results/distance_visualization"):
        os.mkdir("Results/distance_visualization")

    file_path = f"Results/distance_visualization/vid{video}.avi"
    fourcc = cv2.VideoWriter_fourcc("I", "4", "2", "0")
    fps = 20
    frame = (576, 200)
    vid = cv2.VideoWriter(file_path, fourcc, fps, frame)

    distance = distance_evolution(video)

    for t in range(time):
        image_path = f"Data/video{video}/image{t+1}.jpg"
        image = cv2.imread(image_path)
        image_processed = image_processing(image)
        waveform = get_waveform(image_processed)
        distance_variation = distance[t]
        for x in range(width):
            if waveform[x, 0] > 0 and waveform[x, 1] > 0:
                image[waveform[x, 0] : waveform[x, 1], x, :] = float2RGB(
                    distance_variation[x]
                )
                """
                if distance_variation[x] > 0:
                    # change the vertical line to red
                    image[waveform[x, 0]:waveform[x, 1], x, 2] = min(distance_variation[x]*60, 255)
                if distance_variation[x] == 0:
                    # change the vertical line to green
                    image[waveform[x, 0]:waveform[x, 1], x, 1] = 128
                if distance_variation[x] < 0:
                    # change the vertical line to blue
                    image[waveform[x, 0]:waveform[x, 1], x, 0] = min(distance_variation[x]*20, 255)
                """
        vid.write(image)

    vid.release()

    print(f"-- distance visualization of video {video} completed")


def float2RGB(_data):

    _max = 5
    _min = -5
    _range = _max - _min + 1

    r = (_data - _min) / _range
    step = int(_range / 5)
    idx = int(r * 5)
    h = (idx + 1) * step + _min
    m = idx * step + _min
    local_r = (_data - m) / (h - m)

    if _data < _min:
        return np.array([0, 0, 0])
    if _data > _max:
        return np.array([255, 255, 255])
    if idx == 0:
        return np.array([0, int(local_r * 255), 255])
    if idx == 1:
        return np.array([0, 255, int((1 - local_r) * 255)])
    if idx == 2:
        return np.array([int(local_r * 255), 255, 0])
    if idx == 3:
        return np.array([255, int((1 - local_r) * 255), 0])
    if idx == 4:
        return np.array([255, 0, int(local_r * 255)])


def get_waveform(image):
    distance = get_distance(image)
    waveform = np.zeros((width, 2), dtype=np.int16)
    for x in range(width):
        waveform[x, 0] = get_apex_y(image, x)
        waveform[x, 1] = waveform[x, 0] + distance[x]
        # plt.show()
    return waveform


def waveform_evolution(video, output=False):

    waveform = np.zeros((time, width))

    for t in range(time):
        # TODO: how to add a color bar in the animation
        image_path = f"Data/video{video}/image{t+1}.jpg"
        image = cv2.imread(image_path)
        image = image_processing(image)
        waveform[t] = get_waveform(image)[:, 0]

    if not os.path.exists("Results/waveform_evolution"):
        os.mkdir("Results/waveform_evolution")

    if output:

        fig = plt.figure()
        # plt.gca().invert_yaxis()
        plt.title(f"Waveform evolution of video {video}")
        plt.xlabel("coordinate x")
        plt.ylabel("coordinate y")
        plt.ylim((0, 200))

        x = np.arange(width)
        y = waveform[0]
        (line,) = plt.plot(x, y)

        def update(frame):
            line.set_ydata(waveform[frame])
            return (line,)

        # TODO: how to stop the simulation?
        animation = ani.FuncAnimation(
            fig=fig, func=update, frames=np.arange(time), interval=10
        )
        animation.save(f"Results/waveform_evolution/{video}.gif")
        # plt.show()

    print(f"-- waveform evolution of video {video} completed")

    return waveform


def get_spectrum(waveform, output=False):
    waveform -= np.mean(waveform)
    spectrum = np.abs(np.fft.fft(waveform)) / waveform.shape[0] * 2
    if output:
        plt.figure()
        plt.xlabel("frequency")
        plt.ylabel("amplitude")
        plt.scatter(np.arange(mode), spectrum[:mode])
        # plt.show()
    return spectrum


def spectrum_evolution(video, output=False):

    waveform = waveform_evolution(video)
    spectrum = np.zeros((time, mode))

    for t in range(time):
        spectrum[t] = get_spectrum(waveform[t])[:mode]

    if not os.path.exists("Results/spectrum_evolution"):
        os.mkdir("Results/spectrum_evolution")

    if output:
        x = np.arange(time)
        y = np.arange(mode)
        x_grid, y_grid = np.meshgrid(x, y)
        plt.figure()
        plt.title(f"Spectrum evolution of video {video}")
        plt.xlabel("time")
        plt.ylabel("spectrum")
        cm = plt.cm.get_cmap("jet")
        plt.pcolormesh(x_grid, y_grid, spectrum.T, cmap=cm, vmin=0, vmax=35)
        plt.colorbar()
        plt.savefig(f"Results/spectrum_evolution/{video}.png")
        # plt.show()

    print(f"-- spectrum evolution of video {video} completed")

    return spectrum


def get_thickness(image, x_left=30, x_right=530, output=False):

    width_reduced = 540
    x = sp.Symbol("x")
    x_list = np.arange(width)
    x_list_reduced = np.arange(width_reduced)

    def continuous_waveform(waveform, order=8):
        result = 0
        coef = np.polyfit(x_list_reduced, waveform[:width_reduced], order)
        for d in range(len(coef)):
            result += x**d * coef[-d - 1]
        return result

    waveform = get_waveform(image)
    upper_surface = waveform[:, 0]
    lower_surface = waveform[:, 1]
    middle_line = 0.5 * (upper_surface + lower_surface)

    cont_upper_surface = continuous_waveform(upper_surface)
    cont_middle_line = continuous_waveform(middle_line)
    cont_lower_surface = continuous_waveform(lower_surface)

    slope_list = [cont_middle_line.diff(x).subs(x, x_value) for x_value in x_list]
    norm_list = [-1 / slope for slope in slope_list]
    thickness = []

    """
    for x_value in np.linspace(int(x_left), int(x_right), int((x_right-x_left)/10), endpoint=False):
        normal_line = cont_middle_line.subs(x, x_value) + norm_list[int(x_value)]*(x-x_value)
        x_upper_set = sp.solveset(normal_line-cont_upper_surface, x, domain=sp.S.Reals)
        x_lower_set = sp.solveset(normal_line-cont_lower_surface, x, domain=sp.S.Reals)
        x1 = min([abs(x_upper_sol - x_value) for x_upper_sol in x_upper_set])
        x2 = min([abs(x_lower_sol - x_value) for x_lower_sol in x_lower_set])
        thickness.append((x1 + x2) * (1 + norm_list[int(x_value)] ** 2) ** 0.5)
        print(f"    x={x_value}, x1={x1}, x2={x2}, e={thickness[-1]}")
    """

    x_value = x_left
    while x_value <= x_right:
        normal_line = cont_middle_line.subs(x, x_value) + norm_list[int(x_value)] * (
            x - x_value
        )
        x_upper_set = sp.solveset(
            normal_line - cont_upper_surface, x, domain=sp.S.Reals
        )
        x_lower_set = sp.solveset(
            normal_line - cont_lower_surface, x, domain=sp.S.Reals
        )
        x1 = min([abs(x_upper_sol - x_value) for x_upper_sol in x_upper_set])
        x2 = min([abs(x_lower_sol - x_value) for x_lower_sol in x_lower_set])
        thickness.append((x1 + x2) * (1 + norm_list[int(x_value)] ** 2) ** 0.5)
        # print(f"    x={x_value}, x1={x1}, x2={x2}, e={thickness[-1]}")

        # calculate thickness every 10 pixels
        x_value += 10

    if output:
        plt.figure()
        plt.xlabel("coordinate x")
        plt.ylabel("thickness")
        plt.plot(np.arange(len(thickness)), thickness)
        # plt.show()

    return thickness


def get_minimum_thickness(image, x_left=100, x_right=476):

    width_reduced = 540
    x = sp.Symbol("x")
    x_list = np.arange(width)
    x_list_reduced = np.arange(width_reduced)

    def continuous_waveform(waveform, order=8):
        result = 0
        coef = np.polyfit(x_list_reduced, waveform[:width_reduced], order)
        for d in range(len(coef)):
            result += x**d * coef[-d - 1]
        return result

    waveform = get_waveform(image)
    upper_surface = waveform[:, 0]
    lower_surface = waveform[:, 1]
    middle_line = 0.5 * (upper_surface + lower_surface)

    cont_upper_surface = continuous_waveform(upper_surface)
    cont_middle_line = continuous_waveform(middle_line)
    cont_lower_surface = continuous_waveform(lower_surface)

    slope_list = [cont_middle_line.diff(x).subs(x, x_value) for x_value in x_list]
    norm_list = [-1 / slope for slope in slope_list]
    x_min = x_left
    thickness = []
    thickness_min = 30

    x_value = x_left
    while x_value <= x_right:
        normal_line = cont_middle_line.subs(x, x_value) + norm_list[int(x_value)] * (
            x - x_value
        )
        x_upper_set = sp.solveset(
            normal_line - cont_upper_surface, x, domain=sp.S.Reals
        )
        x_lower_set = sp.solveset(
            normal_line - cont_lower_surface, x, domain=sp.S.Reals
        )
        x1 = min([abs(x_upper_sol - x_value) for x_upper_sol in x_upper_set])
        x2 = min([abs(x_lower_sol - x_value) for x_lower_sol in x_lower_set])
        thickness.append((x1 + x2) * (1 + norm_list[int(x_value)] ** 2) ** 0.5)

        # find the minumum of the thickness
        if thickness[-1] < thickness_min:
            thickness_min = thickness[-1]
            x_min = x_value
            print(f"x_min update to {x_min}, thickness_min = {thickness[-1]}")

        x_value += 1

    return x_min


def thickness_evolution(video, x_left, x_right, p_left, p_right):

    x_left = max(x_left, 30)
    x_right = min(x_right, 530)

    thickness = np.zeros((time, int((x_right - x_left) / 10) + 1))

    image_path = f"Data/video{video}/image1.jpg"
    image = cv2.imread(image_path)
    image = image_processing(image)
    thickness0 = np.array(get_thickness(image, x_left=x_left, x_right=x_right))
    x_min = get_minimum_thickness(image, x_left=p_left, x_right=p_right)

    for t in range(19, 101, 1):
        print(f"t = {t}")
        image_path = f"Data/video{video}/image{t+1}.jpg"
        image = cv2.imread(image_path)
        image = image_processing(image)
        thickness[t] = (
            np.array(get_thickness(image, x_left=x_left, x_right=x_right)) - thickness0
        )

    print(f"-- thickness evolution of video {video} completed")

    return thickness, x_min


def thickness_evolution_full(video, output=True):

    thickness = np.zeros((time, 51))

    for t in range(time):
        print(f"t = {t}")
        image_path = f"Data/video{video}/image{t+1}.jpg"
        image = cv2.imread(image_path)
        image = image_processing(image)
        if t == 0:
            thickness[t] = np.array(get_thickness(image))
        else:
            thickness[t] = np.array(get_thickness(image)) - thickness[0]
    thickness[0] = np.zeros(51)

    if not os.path.exists("Results/thickness_evolution"):
        os.mkdir("Results/thickness_evolution")

    if output:
        x = np.arange(time)
        y = np.arange(51)
        x_grid, y_grid = np.meshgrid(x, y)
        plt.figure()
        plt.title(f"Thickness evolution of video {video}")
        plt.xlabel("time")
        plt.ylabel("coordinate x")
        plt.gca().invert_yaxis()
        cm = plt.cm.get_cmap("jet")
        plt.pcolormesh(x_grid, y_grid, thickness.T, cmap=cm, vmin=-5, vmax=5)
        plt.colorbar()
        plt.savefig(f"Results/thickness_evolution/{video}_time.png")
        plt.show()

    print(f"-- thickness evolution of video {video} completed")

    return thickness


def get_length(image):

    width_reduced = 540
    x = sp.Symbol("x")
    x_list = np.arange(width)
    x_list_reduced = np.arange(width_reduced)

    def continuous_waveform(waveform, order=8):
        result = 0
        coef = np.polyfit(x_list_reduced, waveform[:width_reduced], order)
        for d in range(len(coef)):
            result += x**d * coef[-d - 1]
        return result

    waveform = get_waveform(image)
    middle_line = np.mean(waveform, axis=1)

    cont_middle_line = continuous_waveform(middle_line)

    slope_list = [cont_middle_line.diff(x).subs(x, x_value) for x_value in x_list]
    length_list = [(1 + slope**2) ** 0.5 for slope in slope_list]

    return sum(length_list)


def length_evolution(video, output=False):

    length = np.zeros(time)

    for t in range(time):
        print(t, time)
        image_path = f"Data/video{video}/image{t+1}.jpg"
        image = cv2.imread(image_path)
        image = image_processing(image)
        length[t] = get_length(image)

    if not os.path.exists("Results/length_evolution"):
        os.mkdir("Results/length_evolution")

    if output:
        plt.figure()
        plt.title(f"Length evolution of video {video}")
        plt.xlabel("time")
        plt.ylabel("length")
        plt.plot(np.arange(time), length)
        plt.savefig(f"Results/length_evolution/{video}.png")
        # plt.show()

    print(f"-- length evolution of video {video} completed")

    return length


def get_viscosity(video):

    y_list = []

    image_path = f"Data/video{video}/image1.jpg"
    image = cv2.imread(image_path)
    image = image_processing(image)
    apex_x, apex_y = get_apex(image)

    for i in range(100, 139):
        image_path = f"Data/video{video}/image{i+1}.jpg"
        image = cv2.imread(image_path)
        image = image_processing(image)
        # TODO: what should be the right data used here?

        # first case
        """
        y = get_apex_y(image, 0)
        if y > 0:
            y_list.append(y)
        else:
            y_list.append(y_list[-1])
        """
        # second case
        y_list.append(get_apex_y(image, apex_x))

    x_list = np.arange(len(y_list))
    coefficients = np.polyfit(x_list, np.log(y_list), 1)
    # 300000 is the average Young's module (SI units)
    # 0.00023 is the time units
    viscosity = (-coefficients[0] / 300000 / 0.00023) ** (-1)

    print(
        f"-- the viscosity of the cornea in video {video} is {viscosity/1000000:.4f} MPa.s"
    )

    return viscosity


def get_parabola_form(video, output=False):

    image_path = f"Data/video{video}/image1.jpg"
    image = cv2.imread(image_path)
    image = image_processing(image)
    waveform = get_waveform(image, output=False)

    coef = np.polyfit(np.linspace(100, 499, 400), waveform[100:500, 0], 2)
    x = sp.Symbol("x")
    parabola = 0
    for d in range(len(coef)):
        parabola += x**d * coef[-d - 1]
    parabola_list = [
        parabola.subs(x, x_value) for x_value in np.linspace(100, 499, 400)
    ]

    if output:
        plt.figure()
        plt.xlabel("coordinate x")
        plt.ylabel("coordinate y")
        plt.gca().invert_yaxis()
        plt.plot(np.arange(width), waveform[:, 0], label="data")
        plt.plot(np.linspace(100, 499, 400), parabola_list, label="parabola fit")
        plt.legend()
        # plt.show()

    print(f"-- the parabola of the cornea in video {video} is {coef[0]}")

    return coef[0]
