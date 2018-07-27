#ifndef CPPTENSORFLOW_MODEL_LOADER_BASE_H
#define CPPTENSORFLOW_MODEL_LOADER_BASE_H
#include <iostream>
#include <vector>
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
 
using namespace tensorflow;
 
namespace tf_model {
 
/**
 * Base Class for feature adapter, common interface convert input format to tensors
 * */
    class FeatureAdapterBase{
    public:
        FeatureAdapterBase() {};
 
        virtual ~FeatureAdapterBase() {};
 
        virtual void assign(std::string, double[224][224][3],int,int,int) = 0;  // 节点名称和节点数据向量
 
        std::vector<std::pair<std::string, tensorflow::Tensor> > input;
 
    };
 
    class ModelLoaderBase {
    public:
 
        ModelLoaderBase() {};
 
        virtual ~ModelLoaderBase() {};
 
        virtual int load(tensorflow::Session*, const std::string) = 0;     //pure virutal function load method
 
        virtual int predict(tensorflow::Session*, const FeatureAdapterBase&, const std::string, double*) = 0;
 
        tensorflow::GraphDef graphdef; //Graph Definition for current model
 
    };
 
}
 
#endif //CPPTENSORFLOW_MODEL_LOADER_BASE_H
