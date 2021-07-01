# requiered CMAKE version to build the project
cmake_minimum_required (VERSION 3.8)

# current project
project ("PathTracer")

# set comiler flags
# -fopenmp does not work with msvc compiler
if(NOT CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast")
set(CMAKE_CXX_STANDARD 17)

# add external include directories
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/lib/glm") 
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/lib/stb_master")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/lib/raytracing/rt")

# add external libraries
link_directories("${CMAKE_CURRENT_SOURCE_DIR}/lib/glm/glm/lib")
link_directories("${CMAKE_CURRENT_SOURCE_DIR}/lib/raytracing/rt/lib")

# compile and link final executable
add_executable(path_tracer "main.cpp" "rt_app.cpp")

# link external libraries to the executable
target_link_libraries(path_tracer
					  "-fopenmp"
					  "-lglm_static"
					  "-lray_tracing_static")

# additional work
set(CMAKE_EXPORT_COMPILE_COMMANDS on)
