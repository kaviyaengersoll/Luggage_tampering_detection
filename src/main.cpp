#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <onnxruntime_cxx_api.h>
#if USE_DIRECTML
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <dml_provider_factory.h>
#endif
#include "web_server.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <regex>
#include <cctype>
#include <map>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;
struct Config {
    std::string source="0", detector="models/rtmdet_ins_s.onnx", pose="models/rtmpose_m.onnx";
    std::string output="output", dashboard="dashboard", detector_format="mmdeploy_dets_labels";
    std::string detector_output, pose_x_output, pose_y_output;
    int det_w=640, det_h=640, pose_w=192, pose_h=256, person_class=0, bag_class=28;
    int proximity=22, consecutive=2, cooldown=60, inference_interval=12, upload_interval=6;
    bool tile_long_axis=false, upload_precision_mode=false;
    int contact_edge_pixels=24;
    double static_baggage_seconds=3.0;
    float confidence=.25f, person_confidence=.40f;
    float suitcase_confidence=.30f, backpack_confidence=.35f, handbag_confidence=.40f;
    float nms=.5f, pose_confidence=.2f;
};
static bool is_baggage_class(int cls,const Config& c){return cls==c.bag_class||cls==24||cls==26;}
static bool is_baggage_class(int cls,int suitcase_class){return cls==suitcase_class||cls==24||cls==26;}
class OnnxModel {
    Ort::Session session{nullptr};
    std::vector<std::string> inputs,outputs;
public:
    OnnxModel(const Ort::Env& env,const fs::path& file){
        Ort::SessionOptions options;options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        // Half the logical cores keeps inference on physical cores; using all
        // logical cores oversubscribes SMT and measured slower.
        options.SetIntraOpNumThreads(std::max(1,(int)std::thread::hardware_concurrency()/2));
#if USE_DIRECTML
        // DirectML requires sequential execution and disables memory pattern reuse.
        options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        options.DisableMemPattern();
        const OrtDmlApi* dml_api=nullptr;
        Ort::ThrowOnError(Ort::GetApi().GetExecutionProviderApi("DML",ORT_API_VERSION,reinterpret_cast<const void**>(&dml_api)));
        Ort::ThrowOnError(dml_api->SessionOptionsAppendExecutionProvider_DML(options,0));
#endif
#ifdef _WIN32
        session=Ort::Session(env,file.wstring().c_str(),options);
#else
        session=Ort::Session(env,file.string().c_str(),options);
#endif
        Ort::AllocatorWithDefaultOptions allocator;
        for(size_t i=0;i<session.GetInputCount();i++){auto n=session.GetInputNameAllocated(i,allocator);inputs.emplace_back(n.get());}
        for(size_t i=0;i<session.GetOutputCount();i++){auto n=session.GetOutputNameAllocated(i,allocator);outputs.emplace_back(n.get());}
        if(inputs.empty()||outputs.empty())throw std::runtime_error("ONNX model has no inputs or outputs: "+file.string());
    }
    const std::vector<std::string>& output_names()const{return outputs;}
    int find_output(const std::string& hint)const{for(size_t i=0;i<outputs.size();i++)if(outputs[i].find(hint)!=std::string::npos)return (int)i;return -1;}
    std::vector<Ort::Value> run(const cv::Mat& blob){
        if(blob.dims!=4||!blob.isContinuous())throw std::runtime_error("Model input must be a contiguous NCHW tensor");
        std::vector<int64_t> shape={blob.size[0],blob.size[1],blob.size[2],blob.size[3]};
        auto memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
        Ort::Value input=Ort::Value::CreateTensor<float>(memory,const_cast<float*>(blob.ptr<float>()),blob.total(),shape.data(),shape.size());
        std::vector<const char*> input_names,output_names;for(const auto&n:inputs)input_names.push_back(n.c_str());for(const auto&n:outputs)output_names.push_back(n.c_str());
        Ort::RunOptions options;return session.Run(options,input_names.data(),&input,1,output_names.data(),output_names.size());
    }
};
static std::string json_string(const std::string& s,const std::string& key,const std::string& fallback) {
    std::regex re("\\\""+key+"\\\"\\s*:\\s*\\\"([^\\\"]*)\\\""); std::smatch m;
    return std::regex_search(s,m,re)?m[1].str():fallback;
}
static double json_number(const std::string& s,const std::string& key,double fallback) {
    std::regex re("\\\""+key+"\\\"\\s*:\\s*(-?[0-9]+(?:\\.[0-9]+)?)"); std::smatch m;
    return std::regex_search(s,m,re)?std::stod(m[1].str()):fallback;
}
static Config load_config(const std::string& path) {
    std::ifstream f(path); if(!f) throw std::runtime_error("Cannot open config: "+path);
    std::string s((std::istreambuf_iterator<char>(f)),{}); Config c;
#define STR(field,key,def) c.field=json_string(s,key,def)
#define NUM(field,key,def) c.field=static_cast<decltype(c.field)>(json_number(s,key,def))
    STR(source,"source",c.source); STR(detector,"detector_model",c.detector); STR(pose,"pose_model",c.pose);
    STR(output,"output_dir",c.output); STR(dashboard,"dashboard_dir",c.dashboard);
    STR(detector_format,"detector_output_format",c.detector_format); STR(detector_output,"detector_output_name","");
    STR(pose_x_output,"pose_x_output_name",""); STR(pose_y_output,"pose_y_output_name","");
    NUM(det_w,"detector_input_width",640); NUM(det_h,"detector_input_height",640);
    NUM(pose_w,"pose_input_width",192); NUM(pose_h,"pose_input_height",256);
    NUM(person_class,"person_class",0); NUM(bag_class,"suitcase_class",28);
    NUM(proximity,"proximity_pixels",22); NUM(consecutive,"consecutive_frames",2); NUM(cooldown,"cooldown_frames",60);
    NUM(inference_interval,"inference_interval_frames",12); c.inference_interval=std::clamp(c.inference_interval,1,2400);
    NUM(upload_interval,"upload_inference_interval_frames",6); c.upload_interval=std::clamp(c.upload_interval,1,60);
    c.tile_long_axis=std::regex_search(s,std::regex("\\\"tile_long_axis\\\"\\s*:\\s*true",std::regex::icase));
    NUM(contact_edge_pixels,"contact_edge_tolerance_pixels",24); c.contact_edge_pixels=std::clamp(c.contact_edge_pixels,0,32);
    NUM(static_baggage_seconds,"static_baggage_seconds",3.0);
    NUM(confidence,"confidence_threshold",.25); NUM(person_confidence,"person_confidence_threshold",.40);
    NUM(suitcase_confidence,"suitcase_confidence_threshold",.30);
    NUM(backpack_confidence,"backpack_confidence_threshold",.35);
    NUM(handbag_confidence,"handbag_confidence_threshold",.40);
    NUM(nms,"nms_threshold",.5); NUM(pose_confidence,"pose_minimum_confidence",.2);
#undef STR
#undef NUM
    return c;
}
struct Detection { cv::Rect2f box; float score; int cls; int id=-1; cv::Mat mask,mask_distance; cv::Point mask_distance_origin; float mask_opacity=1.f; bool static_background=false,confirmed=false; };
static void build_bag_distance_map(Detection& d,int width,int height){
    d.mask_distance.release();d.mask_distance_origin={};
    if(d.mask.empty()||width<=0||height<=0)return;
    constexpr int padding=34; // exceeds the configured 32 px contact tolerance
    cv::Rect box((int)std::floor(d.box.x),(int)std::floor(d.box.y),(int)std::ceil(d.box.width),(int)std::ceil(d.box.height));
    box&=cv::Rect(0,0,width,height);if(box.empty())return;
    cv::Rect roi(box.x-padding,box.y-padding,box.width+2*padding,box.height+2*padding);
    roi&=cv::Rect(0,0,width,height);if(roi.empty())return;
    cv::Mat inverse;cv::bitwise_not(d.mask(roi),inverse);cv::distanceTransform(inverse,d.mask_distance,cv::DIST_L2,3);d.mask_distance_origin=roi.tl();
}
static float iou(const cv::Rect2f&a,const cv::Rect2f&b){float inter=(a&b).area();return inter/(a.area()+b.area()-inter+1e-6f);}
static float mask_solidity(const cv::Mat& mask){
    std::vector<std::vector<cv::Point>> contours;cv::findContours(mask.clone(),contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);
    if(contours.empty())return 0.f;
    auto largest=std::max_element(contours.begin(),contours.end(),[](const auto&a,const auto&b){return cv::contourArea(a)<cv::contourArea(b);});
    std::vector<cv::Point> hull;cv::convexHull(*largest,hull);const double hull_area=cv::contourArea(hull);
    return hull_area>0?std::min(1.f,(float)cv::countNonZero(mask)/(float)hull_area):0.f;
}
static void standardize_rgb_blob(cv::Mat& blob){
    const float mean[3]={123.675f,116.28f,103.53f},stddev[3]={58.395f,57.12f,57.375f};
    int plane=blob.size[2]*blob.size[3];float* data=blob.ptr<float>();
    for(int c=0;c<3;c++)for(int i=0;i<plane;i++)data[c*plane+i]=(data[c*plane+i]-mean[c])/stddev[c];
}
static void standardize_bgr_blob(cv::Mat& blob){
    const float mean[3]={103.53f,116.28f,123.675f},stddev[3]={57.375f,57.12f,58.395f};
    int plane=blob.size[2]*blob.size[3];float* data=blob.ptr<float>();
    for(int c=0;c<3;c++)for(int i=0;i<plane;i++)data[c*plane+i]=(data[c*plane+i]-mean[c])/stddev[c];
}
static std::vector<Detection> detect_single(OnnxModel& net,const cv::Mat& frame,const Config& c) {
    const float scale=std::min((float)c.det_w/frame.cols,(float)c.det_h/frame.rows);
    int rw=(int)std::round(frame.cols*scale),rh=(int)std::round(frame.rows*scale);
    cv::Mat resized,canvas(c.det_h,c.det_w,CV_8UC3,cv::Scalar(114,114,114));
    cv::resize(frame,resized,{rw,rh}); resized.copyTo(canvas(cv::Rect(0,0,rw,rh)));
    cv::Mat blob=cv::dnn::blobFromImage(canvas,1.0,{c.det_w,c.det_h},{0,0,0},false,false);standardize_bgr_blob(blob);
    auto outs=net.run(blob); if(outs.empty())return {};
    int det_index=c.detector_output.empty()?net.find_output("dets"):net.find_output(c.detector_output);
    int label_index=net.find_output("label"),mask_index=net.find_output("mask");
    if(det_index<0)for(size_t i=0;i<outs.size();i++){auto s=outs[i].GetTensorTypeAndShapeInfo().GetShape();if(!s.empty()&&s.back()==5){det_index=(int)i;break;}}
    if(det_index<0)det_index=0;
    if(c.detector_format=="mmdeploy_dets_labels"){if(label_index<0&&outs.size()>1)label_index=1;if(mask_index<0&&outs.size()>2)mask_index=2;}
    auto det_info=outs[det_index].GetTensorTypeAndShapeInfo();auto det_shape=det_info.GetShape();
    if(det_shape.empty())return {};int cols=(int)det_shape.back(),rows=(int)(det_info.GetElementCount()/cols);const float* data=outs[det_index].GetTensorData<float>();
    const int64_t* labels64=nullptr;const int32_t* labels32=nullptr;const float* labelsf=nullptr;
    if(label_index>=0){auto info=outs[label_index].GetTensorTypeAndShapeInfo();auto type=info.GetElementType();if(type==ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)labels64=outs[label_index].GetTensorData<int64_t>();else if(type==ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)labels32=outs[label_index].GetTensorData<int32_t>();else labelsf=outs[label_index].GetTensorData<float>();}
    std::vector<cv::Rect> boxes; std::vector<float> scores; std::vector<int> classes,source_rows;
    for(int i=0;i<rows;i++) {
        const float* p=data+(size_t)i*cols; int cls=-1; float score=0; float x1,y1,x2,y2;
        if(c.detector_format=="mmdeploy_dets_labels") { if(cols<5||label_index<0)continue;x1=p[0];y1=p[1];x2=p[2];y2=p[3];score=p[4];cls=labels64?(int)labels64[i]:labels32?(int)labels32[i]:labelsf?(int)labelsf[i]:-1; }
        else if(c.detector_format=="rows_xyxy_score_class") { if(cols<6) continue; x1=p[0];y1=p[1];x2=p[2];y2=p[3];score=p[4];cls=(int)p[5]; }
        else if(c.detector_format=="rows_xywh_objectness_classes") {
            if(cols<6) continue; int best=static_cast<int>(std::max_element(p+5,p+cols)-(p+5)); cls=best; score=p[4]*p[5+best];
            x1=p[0]-p[2]/2; y1=p[1]-p[3]/2; x2=p[0]+p[2]/2; y2=p[1]+p[3]/2;
        } else throw std::runtime_error("Unsupported detector_output_format");
        if(cls!=c.person_class&&!is_baggage_class(cls,c))continue;
        // Use the requested size-aware confidence cutoffs for baggage only.
        float threshold=cls==c.person_class?c.person_confidence:0.f;
        float required_score=std::max(c.confidence,threshold);
        if(cls!=c.person_class){
            const float class_threshold=cls==c.bag_class?c.suitcase_confidence:(cls==24?c.backpack_confidence:c.handbag_confidence);
            const float box_area_ratio=std::max(0.f,(x2-x1)*(y2-y1))/((float)c.det_w*c.det_h);
            const float size_threshold=box_area_ratio<.07f?.20f:box_area_ratio<.20f?.25f:.30f;
            required_score=std::max({c.confidence,class_threshold,size_threshold});
        }
        if(score<required_score) continue;
        // Coordinates are expected in detector-input pixels. Undo top-left letterbox.
        x1=std::clamp((x1)/scale,0.f,(float)frame.cols); x2=std::clamp((x2)/scale,0.f,(float)frame.cols);
        y1=std::clamp((y1)/scale,0.f,(float)frame.rows); y2=std::clamp((y2)/scale,0.f,(float)frame.rows);
        if(x2<=x1||y2<=y1) continue;
        boxes.emplace_back(cv::Point((int)x1,(int)y1),cv::Point((int)x2,(int)y2)); scores.push_back(score); classes.push_back(cls);source_rows.push_back(i);
    }
    std::vector<int> keep;
    // Suppress duplicate bag proposals across suitcase, backpack, and handbag
    // classes together; otherwise a single background patch can survive once
    // for each class at the intentionally low small-bag confidence cutoff.
    for(int category:{c.person_class,-1}){std::vector<cv::Rect> cls_boxes;std::vector<float> cls_scores;std::vector<int> indices;
        for(size_t i=0;i<boxes.size();++i)if(category==c.person_class?classes[i]==c.person_class:is_baggage_class(classes[i],c)){cls_boxes.push_back(boxes[i]);cls_scores.push_back(scores[i]);indices.push_back((int)i);}
        const float threshold=category==c.person_class?std::max(c.confidence,c.person_confidence):.20f;
        std::vector<int> local;cv::dnn::NMSBoxes(cls_boxes,cls_scores,threshold,c.nms,local);for(int i:local)keep.push_back(indices[i]);
    }
    std::vector<Detection> result;
    for(int i:keep){Detection d{boxes[i],scores[i],classes[i],-1,{}, {}};
        if(mask_index>=0&&(d.cls==c.person_class||is_baggage_class(d.cls,c))){int mask_row=source_rows[i];auto mask_info=outs[mask_index].GetTensorTypeAndShapeInfo();auto mask_shape=mask_info.GetShape();
            int mask_count=mask_shape.size()>=4&&mask_shape[0]==1?(int)mask_shape[1]:mask_shape.size()>=3?(int)mask_shape[0]:0;
            if(mask_shape.size()>=3&&mask_row<mask_count){int mh=(int)mask_shape[mask_shape.size()-2],mw=(int)mask_shape.back();size_t plane=(size_t)mh*mw;
                const float* mp=outs[mask_index].GetTensorData<float>()+(size_t)mask_row*plane;cv::Mat small_mask(mh,mw,CV_32F,const_cast<float*>(mp));cv::Mat resized_mask;
                cv::resize(small_mask,resized_mask,{c.det_w,c.det_h});cv::Mat full(c.det_h,c.det_w,CV_8U);cv::compare(resized_mask,0.40,full,cv::CMP_GT);
                cv::Rect unpadded(0,0,rw,rh);cv::Mat source;cv::resize(full(unpadded),source,frame.size(),0,0,cv::INTER_NEAREST);
                if(is_baggage_class(d.cls,c)){int kernel=3;cv::morphologyEx(source,source,cv::MORPH_OPEN,cv::getStructuringElement(cv::MORPH_ELLIPSE,{kernel,kernel}));}
                d.mask=source;
            }
        }
        if(is_baggage_class(d.cls,c)&&!d.mask.empty()){
            cv::Rect box((int)std::floor(d.box.x),(int)std::floor(d.box.y),(int)std::ceil(d.box.width),(int)std::ceil(d.box.height));
            box&=cv::Rect(0,0,frame.cols,frame.rows);
            if(box.empty())continue;
            cv::Mat clipped=cv::Mat::zeros(d.mask.size(),CV_8U);d.mask(box).copyTo(clipped(box));d.mask=std::move(clipped);
            const float aspect=(float)box.width/std::max(1,box.height);
            const float mask_fill=(float)cv::countNonZero(d.mask)/std::max(1,box.area());
            const float minimum_mask_fill=c.upload_precision_mode?.10f:.10f;
            if(aspect<.20f||aspect>4.0f||mask_fill<minimum_mask_fill)continue;
            if(c.upload_precision_mode&&box.area()>frame.total()*.60)continue;
            if(c.upload_precision_mode&&box.area()>frame.total()*.12&&d.score<.60f)continue;
            cv::Mat labels,stats,centroids;
            const int component_count=cv::connectedComponentsWithStats(d.mask,labels,stats,centroids,8,CV_32S);
            int largest_label=0,largest_area=0;
            for(int label=1;label<component_count;++label)if(stats.at<int>(label,cv::CC_STAT_AREA)>largest_area){largest_area=stats.at<int>(label,cv::CC_STAT_AREA);largest_label=label;}
            if(largest_label==0)continue;
            cv::Mat largest=cv::Mat::zeros(d.mask.size(),CV_8U);cv::compare(labels,largest_label,largest,cv::CMP_EQ);d.mask=std::move(largest);
            const float minimum_solidity=box.area()>frame.total()*.15?.50f:.45f;
            if((float)largest_area/std::max(1,box.area())<.08f||mask_solidity(d.mask)<minimum_solidity)continue;
            build_bag_distance_map(d,frame.cols,frame.rows);
        }
        result.push_back(std::move(d));
    }
    // Remove suitcase instances whose masks cover almost exactly the same pixels.
    // Box NMS alone can leave these duplicates when the model predicts different boxes.
    std::sort(result.begin(),result.end(),[](const Detection&a,const Detection&b){return a.score>b.score;});
    std::vector<Detection> deduplicated;
    for(auto& candidate:result){bool duplicate=false;
        if(is_baggage_class(candidate.cls,c)&&!candidate.mask.empty())for(const auto& accepted:deduplicated){
            if(!is_baggage_class(accepted.cls,c)||accepted.mask.empty())continue;
            cv::Mat overlap;cv::bitwise_and(candidate.mask,accepted.mask,overlap);
            const int intersection=cv::countNonZero(overlap),smaller=std::min(cv::countNonZero(candidate.mask),cv::countNonZero(accepted.mask));
            if(smaller>0&&(float)intersection/smaller>=.60f){duplicate=true;break;}
        }
        if(!duplicate)deduplicated.push_back(std::move(candidate));
    }
    // Suppress person-shaped false baggage masks while retaining carried cases:
    // a true case mask normally remains distinct from the person's body mask.
    std::vector<Detection> filtered;
    for(auto& candidate:deduplicated){bool on_person=false;
        if(is_baggage_class(candidate.cls,c)&&!candidate.mask.empty()){
            const int mask_area=cv::countNonZero(candidate.mask);
            if(mask_area>0)for(const auto& person:deduplicated)if(person.cls==c.person_class){
                if(!person.mask.empty()){
                    cv::Rect person_box((int)person.box.x,(int)person.box.y,(int)person.box.width,(int)person.box.height);
                    person_box&=cv::Rect(0,0,frame.cols,frame.rows);
                    const auto center_m=cv::moments(candidate.mask,true);
                    if(!person_box.empty()&&center_m.m00>0){
                        const int cx=(int)(center_m.m10/center_m.m00),cy=(int)(center_m.m01/center_m.m00);
                        int inside=0,total=0;
                        for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++){
                            const int x=cx+dx,y=cy+dy;
                            if(x<0||y<0||x>=person.mask.cols||y>=person.mask.rows)continue;
                            ++total;if(person.mask.at<unsigned char>(y,x)!=0)++inside;
                        }
                        const int person_mask_area=cv::countNonZero(person.mask);
                        const float person_box_coverage=(float)cv::countNonZero(candidate.mask(person_box))/mask_area;
                        if(total>0&&inside*2>=total&&person_mask_area>0&&mask_area<person_mask_area*.30f&&
                           person_box_coverage>=.50f&&person.score+.08f>=candidate.score){on_person=true;break;}
                    }
                }
                if(!person.mask.empty()){
                    cv::Mat overlap;cv::bitwise_and(candidate.mask,person.mask,overlap);
                    const int intersection=cv::countNonZero(overlap),person_mask_area=cv::countNonZero(person.mask);
                    const float candidate_coverage=(float)intersection/mask_area;
                    const float person_coverage=person_mask_area>0?(float)intersection/person_mask_area:0.f;
                    const float same_mask_scale=person_mask_area>0?(float)mask_area/person_mask_area:0.f;
                    const bool duplicated_person_mask=candidate_coverage>=.78f&&person_coverage>=.62f&&same_mask_scale>=.62f&&same_mask_scale<=1.45f;
                    const bool person_part_mask=candidate_coverage>=.72f&&mask_area<person.box.area()*.65f;
                    if(duplicated_person_mask||person_part_mask){on_person=true;break;}
                }
            }
        }
        if(!on_person)filtered.push_back(std::move(candidate));
    }
    std::vector<Detection> cleaned;cleaned.reserve(filtered.size());
    for(auto& candidate:filtered){
        if(is_baggage_class(candidate.cls,c)&&!candidate.mask.empty()){
            // Do not subtract an overlapping person mask here. A hand-held or
            // rolling case overlaps its owner's silhouette, and carving the
            // person out dropped exactly the bags involved in tampering. The
            // person-shaped-mask filter above and the contact check below
            // already suppress genuine person/mask duplicates.
            const int original_area=cv::countNonZero(candidate.mask);
            if(original_area==0)continue;
            cv::Mat labels,stats,centroids;
            const int component_count=cv::connectedComponentsWithStats(candidate.mask,labels,stats,centroids,8,CV_32S);
            int largest_label=0,largest_area=0;
            for(int label=1;label<component_count;++label)if(stats.at<int>(label,cv::CC_STAT_AREA)>largest_area){largest_area=stats.at<int>(label,cv::CC_STAT_AREA);largest_label=label;}
            if(largest_label==0)continue;
            cv::Mat largest=cv::Mat::zeros(candidate.mask.size(),CV_8U);cv::compare(labels,largest_label,largest,cv::CMP_EQ);candidate.mask=std::move(largest);
            const float minimum_solidity=candidate.box.area()>frame.total()*.15?.50f:.45f;
            const float minimum_mask_fill=c.upload_precision_mode?.10f:.10f;
            if((float)cv::countNonZero(candidate.mask)/std::max(1.f,candidate.box.area())<minimum_mask_fill||mask_solidity(candidate.mask)<minimum_solidity)continue;
            build_bag_distance_map(candidate,frame.cols,frame.rows);
        }
        cleaned.push_back(std::move(candidate));
    }
    return cleaned;
}
static std::vector<Detection> detect(OnnxModel& net,const cv::Mat& frame,const Config& c) {
    // Wide or tall surveillance clips waste much of the square detector input
    // on padding. Crop along the long axis with overlap so small bags use more
    // detector pixels while objects at crop boundaries remain fully visible.
    const bool vertical=frame.rows>frame.cols;
    const int long_side=vertical?frame.rows:frame.cols;
    const int short_side=vertical?frame.cols:frame.rows;
    // Tiling doubles detector inference for common 16:9 footage. Leave it off
    // by default for faster runs; enable it only when distant bags need more
    // detector pixels.
    if(!c.tile_long_axis||long_side<=short_side*1.35f)return detect_single(net,frame,c);
    const int tile_side=std::min(long_side,std::max(1,(int)std::round(long_side*.62f)));
    const int last_start=long_side-tile_side;
    const int starts[2]={0,last_start};
    std::vector<Detection> candidates;
    for(int start:starts){
        cv::Rect roi=vertical?cv::Rect(0,start,frame.cols,tile_side):cv::Rect(start,0,tile_side,frame.rows);
        auto local=detect_single(net,frame(roi),c);
        for(auto& d:local){
            if(vertical)d.box.y+=(float)start;else d.box.x+=(float)start;
            if(!d.mask.empty()){
                cv::Mat full=cv::Mat::zeros(frame.size(),CV_8U);
                d.mask.copyTo(full(roi));d.mask=std::move(full);
            }
            candidates.push_back(std::move(d));
        }
    }
    std::sort(candidates.begin(),candidates.end(),[](const Detection&a,const Detection&b){return a.score>b.score;});
    auto mask_density=[](const cv::Mat& mask){
        std::vector<cv::Point> points;cv::findNonZero(mask,points);if(points.empty())return 0.f;
        const cv::Rect bounds=cv::boundingRect(points);return (float)points.size()/std::max(1,bounds.area());
    };
    std::vector<Detection> merged;
    for(auto& candidate:candidates){
        bool duplicate=false;
        for(auto& accepted:merged){
            if(candidate.cls==c.person_class&&accepted.cls==c.person_class&&iou(candidate.box,accepted.box)>=.50f){duplicate=true;break;}
            if(is_baggage_class(candidate.cls,c)&&is_baggage_class(accepted.cls,c)){
                if(!candidate.mask.empty()&&!accepted.mask.empty()){
                    cv::Mat overlap;cv::bitwise_and(candidate.mask,accepted.mask,overlap);
                    const int intersection=cv::countNonZero(overlap);
                    const int candidate_area=cv::countNonZero(candidate.mask),accepted_area=cv::countNonZero(accepted.mask);
                    const int smaller=std::min(candidate_area,accepted_area);
                    if(smaller>0&&(float)intersection/smaller>=.35f){
                        const float candidate_density=mask_density(candidate.mask),accepted_density=mask_density(accepted.mask);
                        if(candidate_area<accepted_area&&candidate_density>accepted_density+.05f&&candidate.score+.15f>=accepted.score)accepted=std::move(candidate);
                        duplicate=true;break;
                    }
                } else if(iou(candidate.box,accepted.box)>=.65f){duplicate=true;break;}
            }
        }
        if(!duplicate)merged.push_back(std::move(candidate));
    }
    std::vector<Detection> full_frame;full_frame.reserve(merged.size());
    for(auto& candidate:merged){
        if(is_baggage_class(candidate.cls,c)&&!candidate.mask.empty()){
            const int original_area=cv::countNonZero(candidate.mask);
            // Keep the detector's luggage silhouette intact when a person is
            // carrying or touching it. Person overlap is considered by the
            // contact check; subtracting the person mask here cut real bags.
            const int remaining_area=cv::countNonZero(candidate.mask);
            if(original_area==0||remaining_area<original_area*.20f)continue;
            cv::Mat labels,stats,centroids;
            const int component_count=cv::connectedComponentsWithStats(candidate.mask,labels,stats,centroids,8,CV_32S);
            int largest_label=0,largest_area=0;
            for(int label=1;label<component_count;++label)if(stats.at<int>(label,cv::CC_STAT_AREA)>largest_area){largest_area=stats.at<int>(label,cv::CC_STAT_AREA);largest_label=label;}
            if(largest_label==0||(float)largest_area/std::max(1,original_area)<.20f)continue;
            cv::Mat largest=cv::Mat::zeros(candidate.mask.size(),CV_8U);cv::compare(labels,largest_label,largest,cv::CMP_EQ);candidate.mask=std::move(largest);
            build_bag_distance_map(candidate,frame.cols,frame.rows);
        }
        full_frame.push_back(std::move(candidate));
    }
    return full_frame;
}
// VP8 encoding of a full-resolution frame costs roughly as much as the rest
// of a sample's bookkeeping, so the writer runs on its own thread and the
// pipeline only hands it frames. The bounded queue keeps memory flat when the
// encoder falls behind.
class AsyncVideoWriter {
    cv::VideoWriter writer;
    std::thread worker;
    std::mutex mutex;
    std::condition_variable changed;
    std::deque<cv::Mat> pending;
    bool stopping=false;
    static constexpr size_t max_pending=6;
public:
    bool open(const std::string& path,int codec,double fps,cv::Size size,bool is_color){
        if(!writer.open(path,codec,fps,size,is_color))return false;
        worker=std::thread([this]{
            for(;;){cv::Mat frame;{std::unique_lock<std::mutex> lock(mutex);
                changed.wait(lock,[this]{return stopping||!pending.empty();});
                if(pending.empty()&&stopping)return;
                frame=std::move(pending.front());pending.pop_front();}
                changed.notify_all();writer.write(frame);}
        });
        return true;
    }
    void write(const cv::Mat& frame){
        if(!worker.joinable())return;
        {std::unique_lock<std::mutex> lock(mutex);changed.wait(lock,[this]{return pending.size()<max_pending;});pending.push_back(frame.clone());}
        changed.notify_all();
    }
    void release(){
        if(!worker.joinable())return;
        {std::lock_guard<std::mutex> lock(mutex);stopping=true;}
        changed.notify_all();worker.join();writer.release();
    }
    bool isOpened()const{return writer.isOpened();}
    ~AsyncVideoWriter(){release();}
};
static std::vector<cv::Point2f> pose(OnnxModel& net,const cv::Mat& frame,const Detection& person,const Config& c) {
    cv::Rect r((int)person.box.x,(int)person.box.y,(int)person.box.width,(int)person.box.height);
    r &= cv::Rect(0,0,frame.cols,frame.rows); if(r.empty()) return {};
    const float aspect=(float)c.pose_w/c.pose_h;float cw=(float)r.width,ch=(float)r.height;
    if(cw/ch>aspect)ch=cw/aspect;else cw=ch*aspect;cw*=1.25f;ch*=1.25f;
    cv::Point2f center(r.x+r.width*.5f,r.y+r.height*.5f);cv::Mat crop;
    cv::getRectSubPix(frame,{std::max(1,(int)std::round(cw)),std::max(1,(int)std::round(ch))},center,crop);
    cv::Mat rgb;cv::cvtColor(crop,rgb,cv::COLOR_BGR2RGB);
    cv::Mat blob=cv::dnn::blobFromImage(rgb,1.0,{c.pose_w,c.pose_h},{0,0,0},false,false);standardize_rgb_blob(blob);
    auto outputs=net.run(blob);if(outputs.size()<2)return {};
    int xi=c.pose_x_output.empty()?net.find_output("simcc_x"):net.find_output(c.pose_x_output);
    int yi=c.pose_y_output.empty()?net.find_output("simcc_y"):net.find_output(c.pose_y_output);
    if(xi<0)xi=0;if(yi<0)yi=1;if(xi>=(int)outputs.size()||yi>=(int)outputs.size())return {};
    auto decode=[](Ort::Value& tensor,std::vector<float>& vals,std::vector<float>& conf){
        auto shape=tensor.GetTensorTypeAndShapeInfo().GetShape();if(shape.size()<2)return 0;int k=(int)shape[shape.size()-2],bins=(int)shape.back();
        const float* a=tensor.GetTensorData<float>();vals.resize(k);conf.resize(k);
        for(int i=0;i<k;i++){int best=0;float maxv=-std::numeric_limits<float>::infinity();for(int j=0;j<bins;j++){float v=a[(size_t)i*bins+j];if(v>maxv){maxv=v;best=j;}}vals[i]=(float)best;conf[i]=maxv;}return bins;
    };
    std::vector<float> xs,xc,ys,yc; int xb=decode(outputs[xi],xs,xc),yb=decode(outputs[yi],ys,yc);
    if(xs.size()!=ys.size()) return {}; std::vector<cv::Point2f> points(xs.size(),{-1,-1});
    for(size_t i=0;i<xs.size();i++) if(xc[i]>=c.pose_confidence&&yc[i]>=c.pose_confidence) {
        points[i]={center.x-cw*.5f+(xs[i]/std::max(1,xb-1))*cw,center.y-ch*.5f+(ys[i]/std::max(1,yb-1))*ch};
    } return points;
}
// ByteTrack-style two-stage association: high-score detections first, then low-score
// detections recover unmatched tracks. Unmatched tracks survive a short occlusion.
struct TrackState{int id,cls,missed=0,mask_missed=0,stationary_samples=0,hits=1;bool static_background=false,stationary_anchor_set=false;cv::Point2f stationary_anchor;cv::Rect2f box,mask_reference_box;cv::KalmanFilter kf;cv::Mat mask_patch;
    TrackState(int track_id,int category,cv::Rect2f initial):id(track_id),cls(category),box(initial),kf(8,4,0,CV_32F){
        kf.transitionMatrix=cv::Mat::eye(8,8,CV_32F);for(int i=0;i<4;i++)kf.transitionMatrix.at<float>(i,i+4)=1.f;
        kf.measurementMatrix=cv::Mat::zeros(4,8,CV_32F);for(int i=0;i<4;i++)kf.measurementMatrix.at<float>(i,i)=1.f;
        cv::setIdentity(kf.processNoiseCov,cv::Scalar::all(1e-2));cv::setIdentity(kf.measurementNoiseCov,cv::Scalar::all(1e-1));cv::setIdentity(kf.errorCovPost,cv::Scalar::all(1));
        kf.statePost.at<float>(0)=initial.x+initial.width*.5f;kf.statePost.at<float>(1)=initial.y+initial.height*.5f;kf.statePost.at<float>(2)=initial.width;kf.statePost.at<float>(3)=initial.height;
    }
    void predict(){cv::Mat s=kf.predict();box={s.at<float>(0)-s.at<float>(2)*.5f,s.at<float>(1)-s.at<float>(3)*.5f,std::max(1.f,s.at<float>(2)),std::max(1.f,s.at<float>(3))};}
    void correct(const cv::Rect2f& r){cv::Mat m=(cv::Mat_<float>(4,1)<<r.x+r.width*.5f,r.y+r.height*.5f,r.width,r.height);kf.correct(m);box=r;}
    void remember_mask(const Detection& d){
        if(d.mask.empty())return;
        cv::Rect r((int)std::floor(d.box.x),(int)std::floor(d.box.y),(int)std::ceil(d.box.width),(int)std::ceil(d.box.height));
        r&=cv::Rect(0,0,d.mask.cols,d.mask.rows);if(r.empty())return;
        mask_reference_box=d.box;
        // Keep enough spatial detail for short tracker-carried mask frames.
        // A 64x64 patch made luggage edges visibly coarse during occlusion.
        cv::resize(d.mask(r),mask_patch,{192,192},0,0,cv::INTER_NEAREST);
    }
    cv::Rect2f predicted_mask_box()const{
        cv::Rect2f bounded=box;
        if(mask_reference_box.width>0&&mask_reference_box.height>0){
            bounded.width=std::clamp(bounded.width,mask_reference_box.width*.75f,mask_reference_box.width*1.10f);
            bounded.height=std::clamp(bounded.height,mask_reference_box.height*.75f,mask_reference_box.height*1.10f);
            const float cx=box.x+box.width*.5f,cy=box.y+box.height*.5f;bounded.x=cx-bounded.width*.5f;bounded.y=cy-bounded.height*.5f;
        }
        return bounded;
    }
    cv::Mat restore_mask(int width,int height)const{
        if(mask_patch.empty()||width<=0||height<=0)return {};
        const cv::Rect2f bounded=predicted_mask_box();
        cv::Rect r((int)std::round(bounded.x),(int)std::round(bounded.y),(int)std::round(bounded.width),(int)std::round(bounded.height));
        r&=cv::Rect(0,0,width,height);if(r.empty())return {};
        cv::Mat full(height,width,CV_8U,cv::Scalar(0)),resized;cv::resize(mask_patch,resized,r.size(),0,0,cv::INTER_NEAREST);resized.copyTo(full(r));return full;
    }
};
class ByteTracker{std::vector<TrackState> active;int next_id=0;int max_age=30;float high=.35f,low=.15f,iou_gate=.2f;
public:void update(std::vector<Detection>& ds,int frame_width,int frame_height,int bag_class,double sample_seconds,double static_seconds,int mask_miss_limit,bool upload_precision){
    double suitcase_area_sum=0.0;int suitcase_area_count=0;
    for(const auto&t:active)if(t.cls==bag_class&&t.missed==0&&t.box.area()>0){suitcase_area_sum+=t.box.area();++suitcase_area_count;}
    const double average_suitcase_area=suitcase_area_count?suitcase_area_sum/suitcase_area_count:0.0;
    for(auto&t:active)t.predict();std::vector<int> order(ds.size());std::iota(order.begin(),order.end(),0);
    std::sort(order.begin(),order.end(),[&](int a,int b){return ds[a].score>ds[b].score;});std::vector<char> used(active.size(),0),assigned(ds.size(),0);
    auto pass=[&](bool high_pass){for(int di:order){
        const bool baggage=is_baggage_class(ds[di].cls,bag_class);
        const float high_threshold=upload_precision&&baggage?.30f:high;
        const float low_threshold=upload_precision&&baggage?.20f:low;
        if(assigned[di]||(high_pass?ds[di].score<high_threshold:(ds[di].score<low_threshold||ds[di].score>=high_threshold)))continue;
        float best=iou_gate;int ti=-1;for(size_t t=0;t<active.size();++t)if(!used[t]&&(active[t].cls==ds[di].cls||(is_baggage_class(active[t].cls,bag_class)&&is_baggage_class(ds[di].cls,bag_class)))){
            const auto& track=active[t];float overlap=iou(ds[di].box,track.box);
            if(baggage&&!ds[di].mask.empty()&&!track.mask_patch.empty()&&track.mask_reference_box.width>0&&track.mask_reference_box.height>0){
                // Box overlap alone swaps neighboring suitcases on a crowded belt.
                // Compare the new silhouette with the track's last confirmed mask
                // in the detector's current box before assigning the identity.
                cv::Mat predicted_mask=track.restore_mask(frame_width,frame_height);
                cv::Mat candidate_mask=ds[di].mask;
                if(!predicted_mask.empty()&&candidate_mask.size()==predicted_mask.size()){
                    cv::Mat intersection_mask,union_mask;cv::bitwise_and(candidate_mask,predicted_mask,intersection_mask);cv::bitwise_or(candidate_mask,predicted_mask,union_mask);
                    const int union_area=cv::countNonZero(union_mask);
                    const float mask_iou=union_area>0?(float)cv::countNonZero(intersection_mask)/union_area:0.f;
                    // Treat silhouette overlap as a preference, not a hard
                    // gate: at the 6-frame sampling rate a moving bag can
                    // legitimately have little pixel overlap with its prior
                    // mask, but must still be allowed to match by motion/box.
                    overlap=std::min(1.f,overlap+.35f*mask_iou);
                }
            }
            if(overlap>best){best=overlap;ti=(int)t;}
        }
        if(ti>=0){
            auto& track=active[ti];
            if(is_baggage_class(ds[di].cls,bag_class)){
                const cv::Point2f new_center(ds[di].box.x+ds[di].box.width*.5f,ds[di].box.y+ds[di].box.height*.5f);
                const float min_side=std::max(1.f,std::min(track.box.width,track.box.height));
                const float anchor_tolerance=std::max(8.f,min_side*.08f);
                if(!track.stationary_anchor_set){track.stationary_anchor=new_center;track.stationary_anchor_set=true;track.stationary_samples=1;}
                else if(cv::norm(new_center-track.stationary_anchor)<=anchor_tolerance)++track.stationary_samples;
                else {track.stationary_anchor=new_center;track.stationary_samples=1;track.static_background=false;}
                const double stationary_seconds=std::max(0,track.stationary_samples-1)*sample_seconds;
                const bool larger_than_average=average_suitcase_area>0&&ds[di].box.area()>average_suitcase_area;
                track.static_background=static_seconds>0&&stationary_seconds>static_seconds&&(upload_precision||larger_than_average);
                ds[di].static_background=track.static_background;
            }
            ds[di].id=track.id;ds[di].confirmed=++track.hits>=2;track.correct(ds[di].box);track.missed=0;used[ti]=1;assigned[di]=1;
        }
    }};
    pass(true);pass(false);
    for(size_t i=0;i<ds.size();++i)if(assigned[i])for(auto&t:active)if(t.id==ds[i].id){
        if(is_baggage_class(ds[i].cls,bag_class)&&!ds[i].mask.empty()){t.remember_mask(ds[i]);t.mask_missed=0;}
        else if(is_baggage_class(t.cls,bag_class)&&!t.mask_patch.empty()){
            if(t.mask_missed<mask_miss_limit){++t.mask_missed;ds[i].mask=t.restore_mask(frame_width,frame_height);build_bag_distance_map(ds[i],frame_width,frame_height);}
            else {++t.mask_missed;ds[i].mask.release();ds[i].mask_distance.release();}
        }
        break;
    }
    for(size_t t=0;t<active.size();++t)if(!used[t]){active[t].missed++;if(is_baggage_class(active[t].cls,bag_class)&&!active[t].mask_patch.empty())active[t].mask_missed++;}
    for(size_t i=0;i<ds.size();++i){
        const float new_track_threshold=upload_precision&&is_baggage_class(ds[i].cls,bag_class)?.25f:low;
        if(!assigned[i]&&ds[i].score>=new_track_threshold){ds[i].id=++next_id;active.emplace_back(ds[i].id,ds[i].cls,ds[i].box);if(is_baggage_class(ds[i].cls,bag_class))active.back().remember_mask(ds[i]);}
    }
    // Keep the Kalman-predicted bag mask alive for the track's full short-
    // occlusion window so intermittent detector misses do not blink it away.
    for(const auto&t:active)if(is_baggage_class(t.cls,bag_class)&&t.missed>0&&t.mask_missed>0&&t.mask_missed<=mask_miss_limit&&!t.mask_patch.empty()){
        cv::Mat mask=t.restore_mask(frame_width,frame_height);if(mask.empty())continue;Detection carried{t.predicted_mask_box(),0.f,t.cls,t.id,std::move(mask)};build_bag_distance_map(carried,frame_width,frame_height);carried.mask_opacity=1.f;carried.static_background=t.static_background;carried.confirmed=t.hits>=2;ds.push_back(std::move(carried));
    }
    active.erase(std::remove_if(active.begin(),active.end(),[&](const TrackState&t){return t.missed>max_age;}),active.end());
}};
static float wrist_to_suitcase_distance(cv::Point2f p,const Detection& bag){
    const int x=(int)std::round(p.x)-bag.mask_distance_origin.x,y=(int)std::round(p.y)-bag.mask_distance_origin.y;
    if(!bag.mask_distance.empty()&&x>=0&&y>=0&&x<bag.mask_distance.cols&&y<bag.mask_distance.rows)return bag.mask_distance.at<float>(y,x);
    return std::numeric_limits<float>::max();
}
static bool wrist_overlaps_suitcase_mask(cv::Point2f p,const Detection& bag){
    if(bag.mask.empty()||p.x<0||p.y<0||p.x>=bag.mask.cols||p.y>=bag.mask.rows)return false;
    int x=(int)std::round(p.x),y=(int)std::round(p.y);
    if(bag.mask.at<unsigned char>(y,x)==0)return false;
    // Require the wrist keypoint to land within the mask, not just touch its
    // contour; this is more robust to apparent edge overlap from camera angle.
    int inside=0,total=0;
    for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++){
        int px=x+dx,py=y+dy;
        if(px<0||py<0||px>=bag.mask.cols||py>=bag.mask.rows)continue;
        ++total;if(bag.mask.at<unsigned char>(py,px)!=0)++inside;
    }
    return total>0&&inside*100>=total*78;
}
static cv::Scalar luggage_color(int){return {0,180,255};}
static cv::Point luggage_mask_center(const Detection& bag){auto m=cv::moments(bag.mask,true);if(m.m00>0)return {(int)(m.m10/m.m00),(int)(m.m01/m.m00)};return {(int)(bag.box.x+bag.box.width/2),(int)(bag.box.y+bag.box.height/2)};}
static void draw_luggage_mask(cv::Mat& image,const Detection& bag){
    const cv::Scalar color=luggage_color(bag.id);
    if(bag.mask.empty())return;
    cv::Rect roi((int)std::floor(bag.box.x),(int)std::floor(bag.box.y),(int)std::ceil(bag.box.width),(int)std::ceil(bag.box.height));
    roi&=cv::Rect(0,0,image.cols,image.rows);if(roi.empty())return;
    cv::Mat tint(roi.size(),image.type(),color),blended;
    cv::addWeighted(image(roi),.48,tint,.52,0,blended);blended.copyTo(image(roi),bag.mask(roi));
    std::vector<std::vector<cv::Point>> contours;cv::findContours(bag.mask(roi).clone(),contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);cv::drawContours(image,contours,-1,color,2,cv::LINE_8,cv::noArray(),0,roi.tl());
}
static std::string stamp(double sec){int s=(int)sec,ms=(int)std::round((sec-s)*1000);std::ostringstream o;o<<std::setfill('0')<<std::setw(2)<<s/3600<<":"<<std::setw(2)<<(s/60)%60<<":"<<std::setw(2)<<s%60<<"."<<std::setw(3)<<ms;return o.str();}
struct ContactHistory{std::deque<bool> votes;long last_seen_frame=0,last_sample_frame=0;};
int main(int argc,char**argv) try {
    if(argc>=2&&std::string(argv[1])=="--server"){
        unsigned short port=8081;if(argc>=3){int p=std::stoi(argv[2]);if(p<1||p>65535)throw std::runtime_error("Port must be between 1 and 65535.");port=(unsigned short)p;}
        return run_web_server(port);
    }
    if(argc<2){std::cerr<<"Usage: luggage_monitor <config.json>\n";return 2;}
    Config c=load_config(argv[1]); fs::create_directories(c.output+"/evidence"); fs::create_directories(c.output+"/overlays"); fs::create_directories(c.dashboard);
    Ort::Env ort_env(ORT_LOGGING_LEVEL_WARNING,"luggage_tampering_detection");
    OnnxModel detector(ort_env,c.detector),pose_net(ort_env,c.pose);
    cv::VideoCapture cap; bool numeric=!c.source.empty()&&std::all_of(c.source.begin(),c.source.end(),[](unsigned char ch){return std::isdigit(ch)!=0;});const bool live_source=numeric||c.source.rfind("rtsp://",0)==0||c.source.rfind("rtsps://",0)==0;
    if(!live_source){c.inference_interval=c.upload_interval;c.upload_precision_mode=true;c.suitcase_confidence=std::max(c.suitcase_confidence,.20f);c.backpack_confidence=std::max(c.backpack_confidence,.20f);c.handbag_confidence=std::max(c.handbag_confidence,.25f);}
    if(numeric)cap.open(std::stoi(c.source));else cap.open(c.source,cv::CAP_FFMPEG);
    if(!cap.isOpened())throw std::runtime_error("Unable to open video source: "+c.source);
    bool new_csv=!fs::exists(c.output+"/events.csv")||fs::file_size(c.output+"/events.csv")==0;
    std::ofstream csv(c.output+"/events.csv",std::ios::app); if(new_csv)csv<<"timestamp,frame,person_id,suitcase_id,wrist,distance,image\n";
    double fps=cap.get(cv::CAP_PROP_FPS);if(fps<1||fps>120)fps=25;long frame_no=0;ByteTracker tracker;const long source_frame_count=(long)cap.get(cv::CAP_PROP_FRAME_COUNT);
    const double duration=source_frame_count>0?source_frame_count/fps:0;
    auto write_progress=[&](long frame){std::ofstream f(c.output+"/progress.json",std::ios::trunc);if(f)f<<"{\"current_seconds\":"<<std::fixed<<std::setprecision(3)<<frame/fps<<",\"duration_seconds\":"<<std::max(0.0,duration)<<"}";};
    write_progress(0);std::ofstream annotation_index(c.output+"/annotations.json",std::ios::trunc);
    AsyncVideoWriter annotated_video;bool writer_attempted=false;const std::string annotated_path=c.output+"/annotated.webm";
    std::map<std::pair<int,int>,ContactHistory> contact_history;std::map<int,std::pair<cv::Rect2f,long>> contact_box_history;std::set<std::pair<int,int>> alerted_pairs;
    const fs::path pause_file=fs::path(c.output)/"pause.request";
    while(true){
        while(fs::exists(pause_file)) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        // Keep the same analyzed source frames, but avoid retrieving and
        // materializing the frames skipped by inference sampling.
        bool reached_end=false;
        if(frame_no>0)for(int skipped=0;skipped<c.inference_interval-1;++skipped){
            if(!cap.grab()){reached_end=true;break;}
            ++frame_no;
        }
        if(reached_end)break;
        cv::Mat frame;if(!cap.read(frame)||frame.empty())break;frame_no++;
        if(!writer_attempted){writer_attempted=true;const double output_fps=fps/c.inference_interval;if(!annotated_video.open(annotated_path,cv::VideoWriter::fourcc('V','P','8','0'),output_fps,frame.size(),true))std::cerr<<"Warning: could not open annotated video writer; image preview remains available.\n";}
        auto ds=detect(detector,frame,c);
        // Uploads analyze every sixth source frame; carry a missed luggage mask
        // for at most one analyzed sample to limit stale masks on the belt.
        // Keep the established RTSP tolerance unchanged.
        tracker.update(ds,frame.cols,frame.rows,c.bag_class,(double)c.inference_interval/fps,c.static_baggage_seconds,c.upload_precision_mode?1:2,c.upload_precision_mode);
        std::vector<Detection> people,bags,visible_bags,contact_people;for(auto&d:ds)if(d.cls==c.person_class)people.push_back(d);else if(is_baggage_class(d.cls,c)&&d.confirmed){visible_bags.push_back(d);bags.push_back(d);}
        cv::Mat annotated=frame.clone();
        for(auto&b:bags)if(!b.static_background)draw_luggage_mask(annotated,b);
        std::set<int> contact_bag_ids;
        std::map<int,cv::Point> bag_centers;for(const auto& b:bags)bag_centers[b.id]=luggage_mask_center(b);
        for(auto&p:people){
            bool may_reach_bag=false;
            for(const auto&b:bags){float dx=std::max({0.f,b.box.x-(p.box.x+p.box.width),p.box.x-(b.box.x+b.box.width)});float dy=std::max({0.f,b.box.y-(p.box.y+p.box.height),p.box.y-(b.box.y+b.box.height)});float reach=std::clamp(p.box.height*.35f,24.f,90.f);if(std::hypot(dx,dy)<=reach){may_reach_bag=true;break;}}
            if(!may_reach_bag)continue;
            auto pts=pose(pose_net,frame,p,c);
            for(auto&b:bags){float dist=std::numeric_limits<float>::max();int nearest=-1;bool wrist_overlaps_mask=false;
                const cv::Point bag_center=bag_centers[b.id];
                for(int wrist:{9,10})if((size_t)wrist<pts.size()&&pts[wrist].x>=0){float d=wrist_to_suitcase_distance(pts[wrist],b);if(d<dist){dist=d;nearest=wrist;}
                    wrist_overlaps_mask=wrist_overlaps_mask||wrist_overlaps_suitcase_mask(pts[wrist],b)||d<=c.contact_edge_pixels;
                }
                // Use the person's segmentation mask rather than their broad
                // rectangle to reject masks that mostly belong to a person. A
                // bag center alone can fall on the person when someone leans over it.
                bool bag_mostly_on_person=false;
                if(wrist_overlaps_mask&&!p.mask.empty()&&!b.mask.empty()){
                    cv::Mat overlap;cv::bitwise_and(b.mask,p.mask,overlap);
                    const int bag_pixels=cv::countNonZero(b.mask);
                    bag_mostly_on_person=bag_pixels>0&&(float)cv::countNonZero(overlap)/bag_pixels>=.70f;
                }
                const bool contact_sample=wrist_overlaps_mask&&!bag_mostly_on_person;
                if(b.static_background&&!contact_sample)continue;
                auto key=std::make_pair(p.id,b.id);auto& history=contact_history[key];history.votes.push_back(contact_sample);if(history.votes.size()>3)history.votes.pop_front();history.last_seen_frame=frame_no;history.last_sample_frame=frame_no;
                int contact_votes=(int)std::count(history.votes.begin(),history.votes.end(),true);
                // Use a short majority window to bridge one missed wrist pose
                // sample. The box still requires repeated real contact votes.
                bool contact_confirmed=contact_votes>=c.consecutive;
                if(contact_confirmed){
                    Detection contact_person=p;auto prior_box=contact_box_history.find(p.id);
                    if(prior_box!=contact_box_history.end()&&frame_no-prior_box->second.second<=c.inference_interval*2){
                        constexpr float current_weight=.35f;const auto& old=prior_box->second.first;
                        contact_person.box={old.x*(1-current_weight)+p.box.x*current_weight,old.y*(1-current_weight)+p.box.y*current_weight,old.width*(1-current_weight)+p.box.width*current_weight,old.height*(1-current_weight)+p.box.height*current_weight};
                    }
                    contact_box_history[p.id]={contact_person.box,frame_no};
                    if(std::none_of(contact_people.begin(),contact_people.end(),[&](const Detection& x){return x.id==p.id;}))contact_people.push_back(contact_person);contact_bag_ids.insert(b.id);
                    if(b.static_background)draw_luggage_mask(annotated,b);
                    cv::Rect pr((int)std::round(contact_person.box.x),(int)std::round(contact_person.box.y),(int)std::round(contact_person.box.width),(int)std::round(contact_person.box.height));
                    cv::rectangle(annotated,pr,{30,220,30},2);cv::putText(annotated,"Person "+std::to_string(p.id),pr.tl(),cv::FONT_HERSHEY_SIMPLEX,.6,{30,220,30},2);
                    std::string time=stamp(frame_no/fps),wrist_name=nearest==9?"left":"right";
                    for(int wrist:{9,10})if((size_t)wrist<pts.size()&&pts[wrist].x>=0)cv::circle(annotated,pts[wrist],5,{255,0,0},-1);
                    if(nearest>=0)cv::line(annotated,pts[nearest],bag_center,{0,0,255},2);
                    // Do not let a nearby contact suppress a different confirmed
                    // person/bag pair. The pair set still prevents repeat alerts.
                    if(alerted_pairs.insert({p.id,b.id}).second){std::string file="evidence/event_"+std::to_string(frame_no)+".jpg",full=c.output+"/"+file;cv::imwrite(full,annotated);
                        csv<<time<<","<<frame_no<<","<<p.id<<","<<b.id<<","<<wrist_name<<","<<dist<<","<<file<<"\n";csv.flush();
                        std::ofstream jf(c.output+"/events.json",std::ios::app);jf<<"{\"timestamp\":\""<<time<<"\",\"frame\":"<<frame_no<<",\"person_id\":"<<p.id<<",\"suitcase_id\":"<<b.id<<",\"wrist\":\""<<wrist_name<<"\",\"distance\":"<<dist<<",\"image\":\""<<file<<"\"}\n";std::cout<<"Possible contact at "<<time<<" (person "<<p.id<<", suitcase "<<b.id<<")\n";
                    }
                }
            }
        }
        for(auto it=contact_history.begin();it!=contact_history.end();){
            if(it->second.last_sample_frame!=frame_no){it->second.votes.push_back(false);if(it->second.votes.size()>3)it->second.votes.pop_front();it->second.last_sample_frame=frame_no;}
            if(frame_no-it->second.last_seen_frame>10)it=contact_history.erase(it);else ++it;
        }
        if(!people.empty()||!visible_bags.empty()){
            cv::Mat overlay(frame.size(),CV_8UC4,cv::Scalar(0,0,0,0));
            for(const auto& bag:visible_bags)if(!bag.mask.empty()){
                const float opacity=contact_bag_ids.count(bag.id)?1.f:bag.mask_opacity;
                overlay.setTo(cv::Scalar(0,150,255,115*opacity),bag.mask);
                std::vector<std::vector<cv::Point>> contours;cv::findContours(bag.mask.clone(),contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);
                cv::drawContours(overlay,contours,-1,cv::Scalar(0,150,255,245*opacity),2);
            }
            for(const auto& person:people){
                cv::Rect box((int)std::floor(person.box.x),(int)std::floor(person.box.y),(int)std::ceil(person.box.width),(int)std::ceil(person.box.height));
                box&=cv::Rect(0,0,overlay.cols,overlay.rows);if(box.empty())continue;
                const bool in_contact=std::any_of(contact_people.begin(),contact_people.end(),[&](const Detection& p){return p.id==person.id;});
                if(!in_contact)continue;
                const cv::Scalar color(60,255,60,255);
                cv::rectangle(overlay,box,color,3);
                cv::putText(overlay,"Contact Person "+std::to_string(person.id),box.tl(),cv::FONT_HERSHEY_SIMPLEX,.65,color,2);
            }
            cv::Mat compact;const int overlay_width=std::min(960,overlay.cols);
            if(overlay.cols>overlay_width)cv::resize(overlay,compact,{overlay_width,std::max(1,cvRound((double)overlay.rows*overlay_width/overlay.cols))},0,0,cv::INTER_AREA);else compact=overlay;
            const std::string name="overlay_"+std::to_string(frame_no)+".png";
            if(cv::imwrite(c.output+"/overlays/"+name,compact,{cv::IMWRITE_PNG_COMPRESSION,3}))annotation_index<<"{\"seconds\":"<<std::fixed<<std::setprecision(3)<<frame_no/fps<<",\"overlay\":\"/output/overlays/"<<name<<"\"}\n";
            else annotation_index<<"{\"seconds\":"<<std::fixed<<std::setprecision(3)<<frame_no/fps<<",\"overlay\":\"\"}\n";
        }else annotation_index<<"{\"seconds\":"<<std::fixed<<std::setprecision(3)<<frame_no/fps<<",\"overlay\":\"\"}\n";
                if(annotated_video.isOpened())annotated_video.write(annotated);
        // The live dashboard only needs a preview-sized JPEG. Keep full
        // resolution for inference, evidence images, and the finished video.
        if(live_source){cv::Mat preview;
        if(annotated.cols>960){
            const int preview_height=std::max(1,cvRound((double)annotated.rows*960/annotated.cols));
            cv::resize(annotated,preview,{960,preview_height},0,0,cv::INTER_AREA);
        }else preview=annotated;
        cv::imwrite(c.output+"/latest.jpg",preview,{cv::IMWRITE_JPEG_QUALITY,82});}
        write_progress(frame_no);
    }
    annotated_video.release();annotation_index.close();if(duration>0)write_progress(source_frame_count);
    return 0;
} catch(const cv::Exception&e){std::cerr<<"OpenCV error: "<<e.what()<<"\n";return 1;} catch(const std::exception&e){std::cerr<<"Error: "<<e.what()<<"\n";return 1;}
