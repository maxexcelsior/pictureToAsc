# coding:utf-8
"""
使用 Python 将位图转变为ascii文件
执行方法：通过cmd进入到该py文件所在文件夹，然后执行命令python pic2asc.py --target-image input/图片名称.jpg --grid-size 网格宽 网格高 --cell-size 每个网格边长 --output-file output/输出文件名称.asc
"""

import argparse
import os

import numpy as np
from PIL import Image

# 颜色字典，在这里指定什么颜色要转变为什么数字，冒号左边为要转变成的数字，冒号右边为位图中对应的颜色rgb值
cd = {
    0 : (255, 255, 255), #road
    1 : (255, 234, 0), #plot
    # 2 : (246, 202, 66), #community
    3 : (118, 175, 126), #green
    4 : (250, 0, 0), #main road
    5 : (186, 12, 255), #axis
    6 : (0, 0, 0), #null
}


def splitImage(image, size):
    """
    将图像按网格划分成多个小图像

    @param {Image} image PIL Image 对象
    @param {Tuple[int, int]} size 网格的行数和列数
    @return {List[Image]} 小图像列表
    """

    W, H = image.size[0], image.size[1]  #获取目标图形宽和高的像素数量
    m, n = size  #
    w, h = int(W / m), int(H / n)
    imgs = []
    # 先按行再按列裁剪出 m * n 个小图像
    for j in range(n):
        for i in range(m):
            # 坐标原点在图像左上角
            imgs.append(image.crop((i * w, j * h, (i + 1) * w, (j + 1) * h)))
    return imgs



def getAverageRGB(image):
    """
    计算图像的平均 RGB 值

    将图像包含的每个像素点的 R、G、B 值分别累加，然后除以像素点数，就得到图像的平均 R、G、B
    值

    @param {Image} image PIL Image 对象
    @return {Tuple[int, int, int]} 平均 RGB 值
    """

    # 计算像素点数
    npixels = image.size[0] * image.size[1]
    # 获得图像包含的每种颜色及其计数，结果类似 [(cnt1, (r1, g1, b1)), ...]
    cols = image.getcolors(npixels)
    # 获得每种颜色的 R、G、B 累加值，结果类似 [(c1 * r1, c1 * g1, c1 * g2), ...]
    sumRGB = [(x[0] * x[1][0], x[0] * x[1][1], x[0] * x[1][2]) for x in cols]
    # 分别计算所有颜色的 R、G、B 平均值，算法类似(sum(ci * ri)/np, sum(ci * gi)/np,
    # sum(ci * bi)/np)
    # zip 的结果类似[(c1 * r1, c2 * r2, ..), (c1 * g1, c1 * g2, ...), (c1 * b1,
    # c1 * b2, ...)]
    avg = tuple([int(sum(x) / npixels) for x in zip(*sumRGB)])
    return avg


def getAverageRGBNumpy(image):
    """
    计算图像的平均 RGB 值，使用 numpy 来计算以提升性能

    @param {Image} image PIL Image 对象
    @return {Tuple[int, int, int]} 平均 RGB 值
    """

    # 将 PIL Image 对象转换为 numpy 数据数组
    im = np.array(image)
    # 获得图像的宽、高和深度
    w, h, d = im.shape
    # 将数据数组变形并计算平均值
    return tuple(np.average(im.reshape(w * h, d), axis=0))


def getBestMatchIndex(input_avg):    #input_avg是目标图形的，avgs是替换图像的
    """
    找出颜色值最接近的索引

    把颜色值看做三维空间里的一个点，依次计算目标点跟列表里每个点在三维空间里的距离，从而得到距
    离最近的那个点的索引。

    @param {Tuple[int, int, int]} input_avg 目标颜色值
    @param {List[Tuple[int, int, int]]} avgs 要搜索的颜色值列表
    @return {int} 命中元素的索引
    """
    colors = list(cd.items())

    key = 0
    min_key = 0
    min_dist = float("inf") #正负无穷
    for col in colors:
        # 三维空间两点距离计算公式 (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
        # + (z1 - z2) * (z1 - z2)，这里只需要比较大小，所以无需求平方根值
        dist = (pow((col[1][0] - input_avg[0]), 2) +
                pow((col[1][1] - input_avg[1]), 2) +
                pow((col[1][2]- input_avg[2]), 2))
        if dist < min_dist:  #所有数都比正无穷小，这里相当于获取颜色距离值列表的最小值
            min_dist = dist  
            min_key = key
        key += 1

    return min_key



def createAscii(target_image, grid_size, cellsize, output_file):
    """
    图片马赛克生成

    @param {Image} target_image 目标图像
    @param {List[Image]} input_images 替换图像列表
    @param {Tuple[int, int]} grid_size 网格行数和列数
    @param {bool} reuse_images 是否允许重复使用替换图像
    @return {Image} 马赛克图像
    """

    # 将目标图像切成网格小图像
    print('splitting input image...')
    target_images = splitImage(target_image, grid_size)

    # 为每个网格小图像在替换图像列表里找到颜色最相似的替换图像
    print('building color sequence...')
    color_sequence = []
    # 分 10 组进行，每组完成后打印进度信息，避免用户长时间等待
    count = 0
    batch_size = int(len(target_images) / 10)

    keys = list(cd.keys())


    # 对   每个网格小图像，   从替换颜色列表找到颜色最相似的那个
    for img in target_images:
        # 计算每个小图像的颜色平均值
        avg = getAverageRGB(img)
        # 找到最匹配的那个颜色值
        match_index = getBestMatchIndex(avg)
        color_sequence.append(keys[match_index])
        # 如果完成了一组，打印进度信息
        if count > 0 and batch_size > 10 and count % batch_size == 0:
            print('processed %d of %d...' % (count, len(target_images)))
        count += 1



    lt = [] #构建一个空列表用于装载要输出的数据
    for row in range(grid_size[1]): #遍历每行
        lt.append([]) #在每行中再构建一个新列表用于装载每行的数据
        for col in range(grid_size[0]): #遍历每列
            lt[row].append(str(color_sequence[row*grid_size[0] + col]) + " ") #将la列表中的数据添加到lt列表的对应位置

    print('outputing asc file')
    fi = open(output_file,'w')
    fi.write("ncols" + " " + str(grid_size[0]) + "\n")
    fi.write("nrows" + " " + str(grid_size[1]) + "\n")
    fi.write("xllcorner" + " " + "0.0" + "\n")
    fi.write("yllcorner" + " " + "0.0" + "\n")
    fi.write("cellsize" + " " + str(cellsize) + "\n")
    fi.write("NODATA_value" + " " + "-9999" + "\n")
    for row in range(grid_size[1]): #遍历每行打印数据
        fi.writelines(lt[row])
        fi.write("\n")        
    fi.close()


    return "done."


def main():
    # 定义程序接收的命令行参数
    parser = argparse.ArgumentParser(
        description='Creates a photomosaic from input images')
    parser.add_argument('--target-image', dest='target_image', required=True)
    parser.add_argument('--grid-size', nargs=2,
                        dest='grid_size', required=True)
    parser.add_argument('--cell-size', dest='cellsize', required=True)
    parser.add_argument('--output-file', dest='outfile', required=False)

    # 解析命令行参数
    args = parser.parse_args()

    # 网格大小
    grid_size = (int(args.grid_size[0]), int(args.grid_size[1]))

    #单元大小
    cellsize = int(args.cellsize)

    # 马赛克图像保存路径，默认为 mosaic.png
    output_file = 'output/file.asc'
    if args.outfile:
        output_file = args.outfile

    # 打开目标图像
    print('reading targe image...')
    target_image = Image.open(args.target_image)


    # 生成ascii文件
    print('starting ascii creation...')
    ascii_file = createAscii(target_image, grid_size, cellsize, output_file)

    # 保存马赛克图像
    print(ascii_file)

    print('done.')


if __name__ == '__main__':
    main()
