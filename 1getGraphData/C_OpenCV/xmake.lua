-- xmake.lua

-- 指定 C++ 标准
set_languages("cxx14")

-- 设置编译器
-- set_toolchains("gcc", {"E:/MinGW/mingw64/bin/gcc.exe"}, "g++", {"E:/MinGW/mingw64/bin/g++.exe"})
    -- xmake f -p mingw --sdk=E:/MinGW/mingw64 可以切换到mingw的编译方式

-- 添加可执行文件
target("OpenCV_Test")
    set_kind("binary")
    add_files("displaytest.cpp")

    -- 设置 OpenCV 目录
    -- set_configvar("OpenCV_DIR", "E:/OpenCV/opencv/build/x64/MinGW/install")
    set_configvar("OpenCV_DIR", ".")

    -- 添加 OpenCV 包目录和链接目录
    add_includedirs("${OpenCV_DIR}/include")
    add_linkdirs("${OpenCV_DIR}/lib")

    -- 添加 OpenCV 包
    add_links("opencv_core", "opencv_highgui", "opencv_imgproc")

