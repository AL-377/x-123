import cv2



def draw(face_pos,right_eye,left_eye,nose,in_path,out_path):
    import cv2

    # 读取输入图像
    image = cv2.imread(in_path)

    # 绘制矩形框
    x, y, w, h = face_pos[0],face_pos[1],face_pos[2],face_pos[3]
    lt = (x,y)

    color = (0, 0, 255) # BGR颜色格式

    cv2.rectangle(image, (x,y), (x+w, y+h),color=color, thickness=2)
    cv2.circle(image,lt,radius= 10,color=color,thickness=-1)
    cv2.circle(image, right_eye, radius=5, color=(0,255,0), thickness=-1) # -1表示填充整个圆
    cv2.circle(image, left_eye, radius=5, color=color, thickness=-1)
    cv2.circle(image, nose, radius=5, color=(255,0,0), thickness=-1)


    # 保存输出图像
    cv2.imwrite(out_path, image)

draw([326, 398, 582, 664],(491, 673),(754, 656),(638, 785),"p1.jpeg.jpg","3.png")

