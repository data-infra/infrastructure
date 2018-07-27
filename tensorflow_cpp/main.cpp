#include <iostream>
#include <opencv2/opencv.hpp>    // 这个放在前面 才能不被后面的错误使用
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "ann_model_loader.h"

using namespace cv;
using namespace tensorflow;
using namespace std;

int main(int argc, char* argv[]) {

	//python中代码
	//pil_image = pil_image.resize(_IMAGE_SIZE,Image.ANTIALIAS) - np.array([123, 117, 104])  # 重置图片大小,才能送入到模型入口.pb模型文件的入口为224,224,3
	//pil_image = np.expand_dims(pil_image[:, :, ::-1], axis=0)  # 增加一个维度


	if (argc != 5) {
        	std::cout << "WARNING: Input Args missing" << std::endl;
        	return 0;
    	}
	std::string image_path = argv[1];  //图片文件地址
    	std::string model_path = argv[2];  // pb模型文件地址
        std::string input_tensor_name = argv[3];  // 模型中输入节点的名称   //输入节点名称  会去pb模型中匹配,可以先用load.py查看节点信息
	std::string output_tensor_name = argv[4];  // 模型中输出节点的名称   //输出节点的名称  会去pb模型中匹配,可以先用load.py查看节点信息

	//vsafety/data:0    输入节点名称
	//vsafety/fc2_joint8/fc2_joint8:0    输出节点名称
	//safety_ns.pb的模型输入data    输入节点
	//safety_ns.pb的模型输出prob    输出节点
	//实验模型输入节点名称"input"
	//实验模型输出节点名称"MobilenetV1/Predictions/Reshape_1"

 
	//读入图像
	Mat srcImage=imread(image_path,CV_LOAD_IMAGE_COLOR);   //加载图片, 通道数为3
	if(srcImage.empty())
	{
		printf("can not load image \n");
		return -1;
	}
	std::cout<<"图片维度:"<<srcImage.size<<std::endl;

	Mat dstImage;

	//尺寸调整
	resize(srcImage,dstImage,Size(224,224),0,0,INTER_LINEAR);   // 矩阵变维
	std::cout<<"新图片维度:"<<dstImage.size<<std::endl;
	double data[224][224][3];
	//变成三通道数据,会在模型加载里面变成4通道(224,224,3,1)或者(1,224,224,3)
	for(int row = 0; row < dstImage.rows; row++)
	{
		for(int col = 0; col < dstImage.cols; col++)
		{
			data[row][col][0]=dstImage.at<Vec3b>(row, col)[0]-123;    // 减掉经验值,使得标准化
			data[row][col][1]=dstImage.at<Vec3b>(row, col)[1]-117;    // 减掉经验值,使得标准化
			data[row][col][2]=dstImage.at<Vec3b>(row, col)[2]-104;    // 减掉经验值,使得标准化
		   	// std::cout<<"红:"<<r<<"绿:"<<g<<"蓝:"<<b<<std::endl; 
		}
	 }

	

	//IplImage* imgClr = cvCreateImage(Size(224,224), IPL_DEPTH_8U, 3);
 
    	// 创建新的Session
    	Session* session;
    	Status status = NewSession(SessionOptions(), &session);
    	if (!status.ok()) {
        	std::cout << status.ToString() << "\n";
        	return 0;
    	}
 
    	// 创建预测demo
    	tf_model::ANNModelLoader model;  // 该类用于模型预测
    	if (0 != model.load(session, model_path)) {
        	std::cout << "Error: Model Loading failed..." << std::endl;
        	return 0;
    	}

 
    	// 创建特征向量的转化类,将数据绑定到对应的输入节点上
    	tf_model::ANNFeatureAdapter input_feat;

    	input_feat.assign(input_tensor_name,data,224,224,3);   //将vector向量赋值给网络节点
 
    	// 产生预测结果
    	double prediction[100];
    	if (0 != model.predict(session, input_feat, output_tensor_name, prediction)) {
        	std::cout << "WARNING: Prediction failed..." << std::endl;
    	}
    	std::cout << "Output Prediction Value:" << prediction[0] << std::endl;
 
    	return 0;
}




