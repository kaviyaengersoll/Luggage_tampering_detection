#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>

#include "web_server.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
namespace {
struct RunInfo { std::string source, type, playback; };
fs::path root, out, inbox;
std::mutex run_mutex, status_mutex;
HANDLE child_process = nullptr;
std::atomic<bool> stopping{false};
RunInfo run_info;
std::string run_status = "idle", run_message = "Choose a video or enter an RTSP URL.";
long long status_updated = 0;

std::string json_escape(const std::string& value) {
    std::ostringstream result;
    for (unsigned char c : value) {
        switch (c) {
        case '"': result << "\\\""; break; case '\\': result << "\\\\"; break;
        case '\n': result << "\\n"; break; case '\r': result << "\\r"; break;
        case '\t': result << "\\t"; break;
        default: if (c < 32) result << ' '; else result << (char)c;
        }
    }
    return result.str();
}
std::string quote_json(const std::string& value) { return "\"" + json_escape(value) + "\""; }
std::string lower(std::string s) { std::transform(s.begin(),s.end(),s.begin(),[](unsigned char c){return (char)std::tolower(c);}); return s; }
std::string read_file(const fs::path& p) { std::ifstream f(p,std::ios::binary); return {std::istreambuf_iterator<char>(f),{}}; }
void set_status(const std::string& status,const std::string& message) {
    std::lock_guard<std::mutex> lock(status_mutex);
    run_status=status; run_message=message;
    status_updated=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()/1000;
    fs::create_directories(out);
    std::ofstream f(out/"status.json",std::ios::trunc);
    f << "{\"status\":" << quote_json(run_status) << ",\"message\":" << quote_json(run_message)
      << ",\"updated\":" << status_updated;
    if(!run_info.source.empty()) f << ",\"source\":" << quote_json(run_info.source)
      << ",\"source_type\":" << quote_json(run_info.type) << ",\"playback_url\":" << quote_json(run_info.playback);
    if(status=="completed" && fs::exists(out/"annotated.webm") && fs::file_size(out/"annotated.webm")>0)
        f << ",\"annotated_url\":\"/output/annotated.webm\"";
    f << "}";
}
std::string get_status() {
    std::lock_guard<std::mutex> lock(status_mutex);
    std::string status=read_file(out/"status.json");
    if(status.empty())return "{\"status\":\"idle\",\"message\":\"Choose a video or enter an RTSP URL.\"}";
    std::string progress=read_file(out/"progress.json");
    if(progress.size()>2&&progress.front()=='{'&&progress.back()=='}')status.insert(status.size()-1,","+progress.substr(1,progress.size()-2));
    return status;
}
std::string json_value(const std::string& json,const std::string& key) {
    std::regex re("\\\""+key+"\\\"\\s*:\\s*\\\"((?:\\\\.|[^\\\"])*)\\\""); std::smatch m;
    if(!std::regex_search(json,m,re)) return {};
    std::string s=m[1].str(), result;
    for(size_t i=0;i<s.size();++i) { if(s[i]=='\\' && i+1<s.size()) { char c=s[++i]; if(c=='n')result+='\n';else if(c=='r')result+='\r';else result+=c; } else result+=s[i]; }
    return result;
}
std::string mime_type(const fs::path& p) {
    std::string ext=lower(p.extension().string());
    if(ext==".html")return "text/html; charset=utf-8"; if(ext==".css")return "text/css; charset=utf-8";
    if(ext==".js")return "application/javascript"; if(ext==".json")return "application/json; charset=utf-8";
    if(ext==".jpg"||ext==".jpeg")return "image/jpeg"; if(ext==".png")return "image/png";
    if(ext==".webm")return "video/webm"; if(ext==".mp4"||ext==".m4v")return "video/mp4";
    if(ext==".avi")return "video/x-msvideo"; if(ext==".mov")return "video/quicktime";
    if(ext==".mkv")return "video/x-matroska"; return "application/octet-stream";
}
bool send_all(SOCKET socket,const char* data,size_t size) {
    while(size) { int n=send(socket,data,(int)std::min<size_t>(size,1<<20),0); if(n<=0)return false; data+=n;size-=n; } return true;
}
void respond(SOCKET s,int code,const std::string& type,const std::string& body,const std::vector<std::pair<std::string,std::string>>& extra={}) {
    std::ostringstream h; h<<"HTTP/1.1 "<<code<< (code==200?" OK":code==206?" Partial Content":code==400?" Bad Request":code==404?" Not Found":code==416?" Range Not Satisfiable":" Internal Server Error") <<"\r\nContent-Type: "<<type<<"\r\nContent-Length: "<<body.size()<<"\r\nConnection: close\r\n";
    for(const auto& x:extra)h<<x.first<<": "<<x.second<<"\r\n"; h<<"\r\n"; auto hs=h.str();send_all(s,hs.data(),hs.size());send_all(s,body.data(),body.size());
}
void json_response(SOCKET s,int code,const std::string& body) { respond(s,code,"application/json; charset=utf-8",body); }
std::string request_path(std::string target) { auto q=target.find('?'); if(q!=std::string::npos)target.resize(q); return target; }
std::string url_decode(std::string s) {
    std::string o;for(size_t i=0;i<s.size();++i){if(s[i]=='+')o+=' ';else if(s[i]=='%'&&i+2<s.size()){unsigned v=0;std::istringstream h(s.substr(i+1,2));h>>std::hex>>v;o+=(char)v;i+=2;}else o+=s[i];}return o;
}
std::string header_value(const std::string& headers,const std::string& name) {
    std::istringstream in(headers);std::string line;const auto needle=lower(name)+":";
    while(std::getline(in,line)){if(!line.empty()&&line.back()=='\r')line.pop_back();auto colon=line.find(':');if(colon!=std::string::npos&&lower(line.substr(0,colon))+":"==needle){auto v=line.substr(colon+1);auto a=v.find_first_not_of(" \t");return a==std::string::npos?"":v.substr(a);}}return {};
}
bool recv_exact(SOCKET s,std::ofstream& file,size_t count) {
    char b[65536];while(count){int n=recv(s,b,(int)std::min<size_t>(sizeof(b),count),0);if(n<=0)return false;file.write(b,n);count-=n;}return true;
}
std::string make_id() {
    static std::atomic<unsigned long long> seq{0};std::ostringstream s;s<<std::hex<<GetTickCount64()<<seq++;return s.str();
}
std::string find_project_root() {
    wchar_t path[32768];DWORD n=GetModuleFileNameW(nullptr,path,32768);fs::path p=fs::path(std::wstring(path,n)).parent_path();
    for(int i=0;i<6&&!p.empty();++i){if(fs::exists(p/"config.example.json"))return p.string();p=p.parent_path();}
    return fs::current_path().string();
}
void monitor_child(HANDLE process) {
    WaitForSingleObject(process,INFINITE);DWORD code=1;GetExitCodeProcess(process,&code);CloseHandle(process);
    {std::lock_guard<std::mutex> lock(run_mutex);child_process=nullptr;}
    set_status(stopping.exchange(false)?"stopped":code==0?"completed":"error",code==0?"Processing finished.":"Pipeline stopped with an error. Check output/pipeline.log.");
}
void launch_pipeline(const std::string& source,const std::string& original_filename,bool is_rtsp) {
    std::lock_guard<std::mutex> lock(run_mutex);
    if(child_process){DWORD code;if(GetExitCodeProcess(child_process,&code)&&code==STILL_ACTIVE)throw std::runtime_error("Analysis is already running. Stop it before starting another.");CloseHandle(child_process);child_process=nullptr;}
    fs::path config_path=root/"config.json";if(!fs::exists(config_path))config_path=root/"config.example.json";
    std::string cfg=read_file(config_path);
    for(const auto& model_key:{std::string("detector_model"),std::string("pose_model")}){
        std::string model=json_value(cfg,model_key);if(model.empty())throw std::runtime_error("Missing model path: "+model_key);
        fs::path p=model;if(p.is_relative())p=root/p;if(!fs::exists(p))throw std::runtime_error("Model file not found: "+p.string());
    }
    auto replace_field=[&](const std::string& key,const std::string& val){std::regex re("(\\\""+key+"\\\"\\s*:\\s*\\\")[^\\\"]*(\\\")");std::smatch m;if(std::regex_search(cfg,m,re)){std::string replacement=m[1].str()+json_escape(val)+m[2].str();cfg.replace((size_t)m.position(0),(size_t)m.length(0),replacement);}};
    replace_field("source",source);replace_field("output_dir","output");replace_field("detector_output_format","mmdeploy_dets_labels");
    fs::create_directories(out/"evidence");fs::create_directories(inbox);
    std::error_code ec;fs::remove(out/"pause.request",ec);fs::remove(out/"latest.jpg",ec);fs::remove(out/"annotated.webm",ec);fs::remove(out/"progress.json",ec);fs::remove(out/"annotations.json",ec);fs::remove_all(out/"overlays",ec);fs::create_directories(out/"overlays");
    {std::ofstream ev(out/"events.json",std::ios::trunc);}
    {std::ofstream f(root/"active_config.json",std::ios::trunc);f<<cfg;}
    run_info.source=is_rtsp?"RTSP stream":original_filename;
    run_info.type=is_rtsp?"rtsp":"upload";
    run_info.playback=is_rtsp?"":"/media/"+original_filename;
    stopping=false;
    wchar_t executable_buffer[32768]{};DWORD n=GetModuleFileNameW(nullptr,executable_buffer,(DWORD)std::size(executable_buffer));if(!n||n>=std::size(executable_buffer))throw std::runtime_error("Could not locate the C++ pipeline executable.");fs::path executable(std::wstring(executable_buffer,n));
    std::wstring exe=executable.wstring(),config=(root/"active_config.json").wstring();
    std::wstring command=L"\""+exe+L"\" \""+config+L"\"";
    STARTUPINFOW si{};si.cb=sizeof(si);PROCESS_INFORMATION pi{};
    HANDLE log=CreateFileW((out/"pipeline.log").wstring().c_str(),GENERIC_WRITE,FILE_SHARE_READ|FILE_SHARE_WRITE,nullptr,CREATE_ALWAYS,FILE_ATTRIBUTE_NORMAL,nullptr);
    if(log!=INVALID_HANDLE_VALUE){si.dwFlags|=STARTF_USESTDHANDLES;si.hStdOutput=log;si.hStdError=log;si.hStdInput=GetStdHandle(STD_INPUT_HANDLE);}
    if(!CreateProcessW(exe.c_str(),command.data(),nullptr,nullptr,log!=INVALID_HANDLE_VALUE,CREATE_NO_WINDOW,nullptr,root.wstring().c_str(),&si,&pi)){DWORD e=GetLastError();if(log!=INVALID_HANDLE_VALUE)CloseHandle(log);throw std::runtime_error("Could not start C++ inference process (Windows error "+std::to_string(e)+").");}
    if(log!=INVALID_HANDLE_VALUE)CloseHandle(log);CloseHandle(pi.hThread);child_process=pi.hProcess;
    set_status("running","Processing video…");std::thread(monitor_child,pi.hProcess).detach();
}
void serve_static(SOCKET s,const std::string& path,const std::string& range) {
    fs::path file;
    if(path=="/"||path=="/dashboard/"||path=="/dashboard/index.html")file=root/"dashboard"/"index.html";
    else if(path.rfind("/media/",0)==0){auto name=url_decode(path.substr(7));if(fs::path(name).filename()!=name){respond(s,404,"text/plain","Not found");return;}file=inbox/name;}
    else if(path.rfind("/output/",0)==0){auto rel=url_decode(path.substr(8));file=out/rel;}
    else if(path.rfind("/dashboard/",0)==0){file=root/path.substr(1);}
    else {respond(s,404,"text/plain","Not found");return;}
    std::error_code ec;auto base=(path.rfind("/media/",0)==0?inbox:path.rfind("/output/",0)==0?out:root).lexically_normal();file=fs::weakly_canonical(file,ec);if(ec||!fs::exists(file)){respond(s,404,"text/plain","Not found");return;}
    auto rel=file.lexically_relative(fs::weakly_canonical(base));if(rel.empty()||rel.native().rfind(L"..",0)==0){respond(s,404,"text/plain","Not found");return;}
    size_t size=(size_t)fs::file_size(file),start=0,end=size?size-1:0;int status=200;
    if(!range.empty()){
        std::smatch m;if(!std::regex_match(range,m,std::regex("bytes=(\\d*)-(\\d*)"))||size==0){respond(s,416,"text/plain","",{{"Content-Range","bytes */"+std::to_string(size)}});return;}
        if(m[1].str().empty()){size_t suffix=std::stoull(m[2].str());start=suffix>=size?0:size-suffix;}else start=std::stoull(m[1].str());
        end=m[2].str().empty()?size-1:std::min(size-1,(size_t)std::stoull(m[2].str()));if(start>=size||end<start){respond(s,416,"text/plain","",{{"Content-Range","bytes */"+std::to_string(size)}});return;}status=206;
    }
    std::ostringstream h;h<<"HTTP/1.1 "<<(status==206?"206 Partial Content":"200 OK")<<"\r\nContent-Type: "<<mime_type(file)<<"\r\nContent-Length: "<<(end-start+1)<<"\r\nAccept-Ranges: bytes\r\nConnection: close\r\n";
    if(status==206)h<<"Content-Range: bytes "<<start<<"-"<<end<<"/"<<size<<"\r\n";h<<"\r\n";auto hs=h.str();if(!send_all(s,hs.data(),hs.size()))return;
    std::ifstream f(file,std::ios::binary);f.seekg((std::streamoff)start);char b[65536];size_t remain=end-start+1;while(remain&&f){size_t n=std::min(sizeof(b),remain);f.read(b,(std::streamsize)n);auto got=(size_t)f.gcount();if(!got)break;if(!send_all(s,b,got))break;remain-=got;}
}
void handle_client(SOCKET s) {
    std::string header;char b[8192];size_t body_already=0;int n;
    while(header.find("\r\n\r\n")==std::string::npos&&header.size()<65536){n=recv(s,b,sizeof(b),0);if(n<=0){closesocket(s);return;}header.append(b,n);}
    auto split=header.find("\r\n\r\n");if(split==std::string::npos){respond(s,400,"text/plain","Bad request");closesocket(s);return;}
    std::string heads=header.substr(0,split),body=header.substr(split+4);body_already=body.size();std::istringstream first(heads);std::string method,target,version;first>>method>>target>>version;std::string path=request_path(target);
    size_t length=0;try{length=(size_t)std::stoull(header_value(heads,"Content-Length"));}catch(...){}
    if(method=="GET"||method=="HEAD"){
        if(path=="/api/status") {auto value=get_status();respond(s,200,"application/json; charset=utf-8",value);}
        else if(method=="HEAD"){serve_static(s,path,header_value(heads,"Range"));}
        else serve_static(s,path,header_value(heads,"Range"));
    } else if(method=="POST"&&path=="/api/upload") {
        if(!length||length>1024ull*1024*1024+65536){json_response(s,400,"{\"error\":\"Choose a video smaller than 1 GB.\"}");}
        else {
            std::string ct=header_value(heads,"Content-Type");std::smatch m;
            if(!std::regex_search(ct,m,std::regex("boundary=(?:\\\"([^\\\"]+)\\\"|([^; ]+))"))){json_response(s,400,"{\"error\":\"Upload request is missing its file boundary.\"}");}
            else {
                std::string boundary=m[1].matched?m[1].str():m[2].str();fs::path temp=out/("upload_"+make_id()+".multipart");fs::create_directories(out);
                {std::ofstream f(temp,std::ios::binary);if(!f){json_response(s,500,"{\"error\":\"Could not create temporary upload file.\"}");closesocket(s);return;}if(body_already){size_t take=std::min(body_already,length);f.write(body.data(),take);}if(body_already<length&&!recv_exact(s,f,length-body_already)){f.close();fs::remove(temp);json_response(s,400,"{\"error\":\"Upload ended before the file was received.\"}");closesocket(s);return;}}
                std::ifstream prefix_file(temp,std::ios::binary);std::string prefix(32768,'\0');prefix_file.read(prefix.data(),(std::streamsize)prefix.size());prefix.resize((size_t)prefix_file.gcount());prefix_file.close();auto filename_at=prefix.find("filename=\"");auto sep=prefix.find("\r\n\r\n");auto close_marker=std::string("\r\n--")+boundary+"--";
                if(filename_at==std::string::npos||sep==std::string::npos){fs::remove(temp);json_response(s,400,"{\"error\":\"No video file was received.\"}");}
                else {
                    filename_at+=10;auto filename_end=prefix.find('"',filename_at);std::string original=filename_end==std::string::npos?"":fs::path(prefix.substr(filename_at,filename_end-filename_at)).filename().string();auto dot=original.find_last_of('.');std::string ext=dot==std::string::npos?"":lower(original.substr(dot));
                    if(ext!=".mp4"&&ext!=".mov"&&ext!=".mkv"&&ext!=".avi"&&ext!=".webm"&&ext!=".m4v"){fs::remove(temp);json_response(s,400,"{\"error\":\"Choose an MP4, MOV, MKV, AVI, WebM, or M4V video.\"}");}
                    else {size_t data_start=sep+4, file_size=(size_t)fs::file_size(temp), trailer=close_marker.size()+2;if(file_size<data_start+trailer){fs::remove(temp);json_response(s,400,"{\"error\":\"Malformed video upload.\"}");closesocket(s);return;}size_t data_size=file_size-data_start-trailer;fs::path dest=inbox/(make_id()+ext);std::ifstream in(temp,std::ios::binary);in.seekg((std::streamoff)data_start);std::ofstream out_file(dest,std::ios::binary);char chunk[65536];size_t remain=data_size;while(remain&&in){size_t take=std::min(sizeof(chunk),remain);in.read(chunk,(std::streamsize)take);size_t got=(size_t)in.gcount();if(!got)break;out_file.write(chunk,(std::streamsize)got);remain-=got;}out_file.close();in.close();fs::remove(temp);
                        if(remain){fs::remove(dest);json_response(s,400,"{\"error\":\"Video upload was incomplete.\"}");}
                        else try{launch_pipeline(dest.string(),dest.filename().string(),false);json_response(s,200,"{\"ok\":true}");}catch(const std::exception&e){json_response(s,400,"{\"error\":"+quote_json(e.what())+"}");}
                    }
                }
            }
        }
    } else if(method=="POST") {
        if(length>65536){json_response(s,400,"{\"error\":\"Request is too large.\"}");}
        else {if(body.size()<length){body.resize(length);size_t got=body_already;while(got<length){n=recv(s,body.data()+got,(int)(length-got),0);if(n<=0)break;got+=n;}body.resize(got);}try{
            if(path=="/api/rtsp"){std::string url=json_value(body,"url");auto l=lower(url);if(l.rfind("rtsp://",0)!=0&&l.rfind("rtsps://",0)!=0)throw std::runtime_error("Enter a valid rtsp:// or rtsps:// URL.");launch_pipeline(url,"",true);json_response(s,200,"{\"ok\":true}");}
            else if(path=="/api/pause"||path=="/api/resume"){{std::lock_guard<std::mutex> lock(run_mutex);if(!child_process)throw std::runtime_error("There is no active analysis.");if(path=="/api/pause")std::ofstream(out/"pause.request").put('1');else{std::error_code ec;fs::remove(out/"pause.request",ec);}}set_status(path=="/api/pause"?"paused":"running",path=="/api/pause"?"Analysis paused.":"Analysis resumed.");json_response(s,200,path=="/api/pause"?"{\"ok\":true,\"status\":\"paused\"}":"{\"ok\":true}");}
            else if(path=="/api/stop"){
                HANDLE wait_handle=nullptr;
                {std::lock_guard<std::mutex> lock(run_mutex);if(child_process){stopping=true;set_status("stopping","Stopping analysis…");TerminateProcess(child_process,1);DuplicateHandle(GetCurrentProcess(),child_process,GetCurrentProcess(),&wait_handle,0,FALSE,DUPLICATE_SAME_ACCESS);}}
                if(wait_handle){WaitForSingleObject(wait_handle,5000);CloseHandle(wait_handle);}
                std::error_code ec;
                for(const auto& name:{"events.json","events.csv","events.sqlite","latest.jpg","annotated.webm","annotations.json","progress.json","pipeline.log"})fs::remove(out/name,ec);
                fs::remove_all(out/"evidence",ec);fs::create_directories(out/"evidence",ec);fs::remove_all(out/"overlays",ec);
                for(const auto& entry:fs::directory_iterator(out,ec))if(entry.is_regular_file()&&entry.path().filename().string().rfind("events_archive_",0)==0&&entry.path().extension()==".json")fs::remove(entry.path(),ec);
                {std::lock_guard<std::mutex> lock(status_mutex);run_info={};run_status="idle";run_message="Alerts and records cleared.";status_updated=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()/1000;std::ofstream f(out/"status.json",std::ios::trunc);f<<"{\"status\":\"idle\",\"message\":\"Alerts and records cleared.\",\"updated\":"<<status_updated<<"}";}
                json_response(s,200,"{\"ok\":true}");
            }
            else json_response(s,404,"{\"error\":\"Not found\"}");
        }catch(const std::exception&e){json_response(s,400,"{\"error\":"+quote_json(e.what())+"}");}}
    } else json_response(s,404,"{\"error\":\"Not found\"}");
    closesocket(s);
}
}

int run_web_server(unsigned short port) {
    root=find_project_root();out=root/"output";inbox=root/"input";fs::create_directories(out);fs::create_directories(inbox);
    fs::path old_events=out/"events.json";
    if(fs::exists(old_events)&&fs::file_size(old_events)>0){std::error_code archive_error;fs::rename(old_events,out/("events_archive_"+make_id()+".json"),archive_error);if(archive_error)std::cerr<<"Warning: could not archive previous alert list: "<<archive_error.message()<<"\n";}
    {std::ofstream fresh_events(out/"events.json",std::ios::trunc);}
    run_info={};set_status("idle","Choose a video or enter an RTSP URL.");
    WSADATA wsa{};if(WSAStartup(MAKEWORD(2,2),&wsa)!=0){std::cerr<<"Winsock initialization failed.\n";return 1;}
    SOCKET server=socket(AF_INET,SOCK_STREAM,IPPROTO_TCP);if(server==INVALID_SOCKET){WSACleanup();return 1;}
    BOOL reuse=TRUE;setsockopt(server,SOL_SOCKET,SO_REUSEADDR,(const char*)&reuse,sizeof(reuse));sockaddr_in addr{};addr.sin_family=AF_INET;addr.sin_port=htons(port);addr.sin_addr.s_addr=htonl(INADDR_LOOPBACK);
    if(bind(server,(sockaddr*)&addr,sizeof(addr))==SOCKET_ERROR||listen(server,16)==SOCKET_ERROR){std::cerr<<"Could not listen on 127.0.0.1:"<<port<<" (Windows error "<<WSAGetLastError()<<").\n";closesocket(server);WSACleanup();return 1;}
    std::cout<<"Luggage Tampering Detection (C++): http://127.0.0.1:"<<port<<"\n";
    while(true){SOCKET client=accept(server,nullptr,nullptr);if(client==INVALID_SOCKET)continue;std::thread(handle_client,client).detach();}
}
