#ifndef CPPTENSORFLOW_ANN_MODEL_LOADER_H
#define CPPTENSORFLOW_ANN_MODEL_LOADER_H
 
#include "model_loader_base.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
 
using namespace tensorflow;
 
namespace tf_model {
 
/**
 * @brief: Model Loader for Feed Forward Neural Network
 * */
    class ANNFeatureAdapter: public FeatureAdapterBase {
    public:
 
        ANNFeatureAdapter();
 
        ~ANNFeatureAdapter();
 
        void assign(std::string tname, double data[224][224][3],int width,int height,int deep) override; // (tensor_name, tensor)
 
    };
 
    class ANNModelLoader: public ModelLoaderBase {
    public:
        ANNModelLoader();
 
        ~ANNModelLoader();
 
        int load(tensorflow::Session*, const std::string) override;    //Load graph file and new session
 
        int predict(tensorflow::Session*, const FeatureAdapterBase&, const std::string, double*) override;
 
    };
 
}
 
 
#endif //CPPTENSORFLOW_ANN_MODEL_LOADER_H
