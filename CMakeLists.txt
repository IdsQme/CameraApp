cmake_minimum_required(VERSION 3.10)


# Set project name
project(CameraDepthPerception)

# Find all .cpp files in the src folder
file(GLOB SOURCES "src/*.cpp")

# Create the executable using all the .cpp files found
add_executable(CameraDepthPerception ${SOURCES})

# Find OpenCV package
set(OpenCV_ROOT "C:/Users/Qme/vcpkg/installed/x64-windows/share/opencv4")
find_package(OpenCV REQUIRED)

# Link OpenCV to your target
target_link_libraries(CameraDepthPerception PRIVATE ${OpenCV_LIBS})

# Include OpenCV headers
target_include_directories(CameraDepthPerception PRIVATE ${OpenCV_INCLUDE_DIRS})

# Include your project's headers (include directory)
target_include_directories(CameraDepthPerception PRIVATE ${CMAKE_SOURCE_DIR}/include)

