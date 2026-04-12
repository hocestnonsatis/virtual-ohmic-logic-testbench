cmake_minimum_required(VERSION 3.14)
project(VOLT CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
  add_compile_options(/O2 /W4 /EHsc)
else()
  add_compile_options(-O2 -Wall -Wextra)
endif()

set(VOLT_SOURCES
    src/dac.cpp
    src/adc.cpp
    src/crossbar.cpp
    src/noise.cpp
    src/activation.cpp
    src/benchmark.cpp
    src/config_json.cpp
    src/weights_csv.cpp
)

add_executable(volt ${VOLT_SOURCES} src/main.cpp)
target_include_directories(volt PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

function(add_volt_test name source)
    add_executable(${name} ${source} ${VOLT_SOURCES})
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

add_volt_test(test_core tests/test_core.cpp)
add_volt_test(test_equivalence tests/test_equivalence.cpp)

enable_testing()
add_test(NAME volt_benchmark COMMAND $<TARGET_FILE:volt> --benchmark)
set_tests_properties(volt_benchmark PROPERTIES WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

add_test(NAME volt_config COMMAND $<TARGET_FILE:volt> --config ${CMAKE_SOURCE_DIR}/volt.example.json)
set_tests_properties(volt_config PROPERTIES WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

add_test(NAME volt_weights COMMAND $<TARGET_FILE:volt> --weights ${CMAKE_SOURCE_DIR}/volt.example.weights.csv --inputs ${CMAKE_SOURCE_DIR}/volt.example.inputs.csv)
set_tests_properties(volt_weights PROPERTIES WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
