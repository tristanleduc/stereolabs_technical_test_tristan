find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h PATHS /Users/tristanleduc/Downloads/onnxruntime/include)
find_library(ONNXRUNTIME_LIBRARY onnxruntime HINTS /Users/tristanleduc/Downloads/onnxruntime/lib)

if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
    set(ONNXRUNTIME_FOUND TRUE)
    set(ONNXRUNTIME_LIBRARIES ${ONNXRUNTIME_LIBRARY})
    set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_INCLUDE_DIR})
else()
    set(ONNXRUNTIME_FOUND FALSE)
endif()

mark_as_advanced(ONNXRUNTIME_INCLUDE_DIR ONNXRUNTIME_LIBRARY)