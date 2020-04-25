import os
import tensorflow as tf


def first(filename):
    # print(type(filename))
    file1_name = filename+"/1.png"
    file2_name = filename + "/2.png"
    file3_name = filename + "/3.png"
    file4_name = filename + "/4.png"
    file5_name = filename + "/5.png"
    image1 = convert_image(file1_name)
    image2 = convert_image(file2_name)
    image3 = convert_image(file3_name)
    image4 = convert_image(file4_name)
    image5 = convert_image(file5_name)
    image = tf.concat([image1,image2],2)
    image = tf.concat([image, image3], 2)
    image = tf.concat([image, image4], 2)
    # labe =  convert_image(label)
    # label = image5
    return image

def first1(filename):
    # print(type(filename))
    # file1_name = filename+"/1.png"
    # file2_name = filename + "/2.png"
    # file3_name = filename + "/3.png"
    # file4_name = filename + "/4.png"
    file5_name = filename + "/5.png"
    # image1 = convert_image(file1_name)
    # image2 = convert_image(file2_name)
    # image3 = convert_image(file3_name)
    # image4 = convert_image(file4_name)
    image5 = convert_image(file5_name)
    # image = tf.concat([image1,image2],2)
    # image = tf.concat([image, image3], 2)
    # image = tf.concat([image, image4], 2)
    # labe =  convert_image(label)
    label = image5
    return label


def convert_image(name):
    imagae = tf.io.read_file(name)
    image = tf.image.decode_png(imagae, channels=1)  # 如果是3通道就改成3
    image = tf.image.resize(image, [1024, 1024])
    image /= 255.0


    return image

if __name__ == '__main__':

    filename = os.listdir("./speckle1")
    print(filename)
    file_list = [os.path.join("./speckle1/",file) for file in filename ]
    print(file_list)
    filenames = tf.constant(file_list)
    labels = tf.constant(list(range(1,81,1)))

    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset_out = tf.data.Dataset.from_tensor_slices(filenames)
    dataset_out = dataset_out.map(first1)

    dataset = dataset.map(first)
    print(dataset)
    dataset = dataset.map(tf.io.serialize_tensor)
    dataset_out = dataset_out.map(tf.io.serialize_tensor)
    ###序列化dataset
    # datsets = dataset.map(tf.io.serialize_tensor)
    # ###写序列化后的dataset成TF文件
    tfrec = tf.data.experimental.TFRecordWriter('input.tfrec')
    tfrec.write(dataset)##写完input
    tfrec = tf.data.experimental.TFRecordWriter('output.tfrec')
    tfrec.write(dataset_out)##写完output
