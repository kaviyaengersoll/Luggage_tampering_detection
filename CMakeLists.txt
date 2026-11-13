cmake_minimum_required(VERSION 3.20)
project(luggage_tampering_detection LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio dnn)

option(ENABLE_DIRECTML "Use ONNX Runtime DirectML on supported Windows GPUs" OFF)
if(WIN32 AND ENABLE_DIRECTML AND EXISTS "${CMAKE_SOURCE_DIR}/deps/onnxruntime-directml-1.24.4/build/native/include/onnxruntime_cxx_api.h")
  set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/deps/onnxruntime-directml-1.24.4" CACHE PATH "ONNX Runtime package root" FORCE)
  set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/build/native/include")
  set(ONNXRUNTIME_LIBRARY "${ONNXRUNTIME_ROOT}/runtimes/win-x64/native/onnxruntime.lib")
  set(ONNXRUNTIME_DLL_DIR "${ONNXRUNTIME_ROOT}/runtimes/win-x64/native")
  set(USE_DIRECTML 1)
else()
  set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/deps/onnxruntime-win-x64-1.31.0" CACHE PATH "ONNX Runtime C/C++ package root")
  find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h PATHS "${ONNXRUNTIME_ROOT}/include" REQUIRED NO_DEFAULT_PATH)
  find_library(ONNXRUNTIME_LIBRARY onnxruntime PATHS "${ONNXRUNTIME_ROOT}/lib" REQUIRED NO_DEFAULT_PATH)
  set(ONNXRUNTIME_DLL_DIR "${ONNXRUNTIME_ROOT}/lib")
  set(USE_DIRECTML 0)
endif()

add_executable(luggage_monitor src/main.cpp src/web_server.cpp)
target_link_libraries(luggage_monitor PRIVATE ${OpenCV_LIBS} "${ONNXRUNTIME_LIBRARY}" ws2_32)
target_include_directories(luggage_monitor PRIVATE ${OpenCV_INCLUDE_DIRS} "${ONNXRUNTIME_INCLUDE_DIR}")
target_compile_definitions(luggage_monitor PRIVATE USE_DIRECTML=${USE_DIRECTML})
add_custom_command(TARGET luggage_monitor POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${ONNXRUNTIME_DLL_DIR}/onnxruntime.dll" "$<TARGET_FILE_DIR:luggage_monitor>/onnxruntime.dll"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${ONNXRUNTIME_DLL_DIR}/onnxruntime_providers_shared.dll" "$<TARGET_FILE_DIR:luggage_monitor>/onnxruntime_providers_shared.dll")
# Keep the OpenCV runtime beside the executable so it starts even when launched
# from Explorer or a shortcut whose PATH does not include the OpenCV bin folder.
set(OPENCV_RUNTIME_DIR "${OpenCV_DIR}/../bin")
file(GLOB OPENCV_RUNTIME_DLLS
  "${OPENCV_RUNTIME_DIR}/opencv_world${OpenCV_VERSION_MAJOR}${OpenCV_VERSION_MINOR}${OpenCV_VERSION_PATCH}.dll"
  "${OPENCV_RUNTIME_DIR}/opencv_videoio_ffmpeg${OpenCV_VERSION_MAJOR}${OpenCV_VERSION_MINOR}${OpenCV_VERSION_PATCH}_64.dll")
foreach(OPENCV_RUNTIME_DLL IN LISTS OPENCV_RUNTIME_DLLS)
  add_custom_command(TARGET luggage_monitor POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${OPENCV_RUNTIME_DLL}" "$<TARGET_FILE_DIR:luggage_monitor>/")
endforeach()
if(MSVC)
  target_compile_options(luggage_monitor PRIVATE /W4 /permissive-)
else()
  target_compile_options(luggage_monitor PRIVATE -Wall -Wextra -Wpedantic)
endif()
