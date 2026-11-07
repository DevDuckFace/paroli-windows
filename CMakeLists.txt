cmake_minimum_required(VERSION 3.15)
project(paroli LANGUAGES CXX)

# paroli for Windows (and Linux): streaming Piper TTS.
#
# Windows: dependencies come from vcpkg (manifest mode, see vcpkg.json) and
# piper-phonemize (espeak-ng + onnxruntime 1.14.1) from its official
# Windows release, downloaded by tools/fetch-deps.ps1 into deps/.
#
#   cmake -B build -DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%/scripts/buildsystems/vcpkg.cmake
#   cmake --build build --config Release
#
# Targets: paroli-cli, paroli-stream (stdin/stdout streaming, no server deps)
# and paroli-server (Drogon HTTP/WebSocket, PAROLI_BUILD_SERVER=ON).

option(USE_RKNN "Enable RKNN for accelerated inference" OFF)
option(PAROLI_BUILD_SERVER "Build paroli-server (needs Drogon, Opus, soxr)" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

if (WIN32)
    # piper-phonemize's Windows release already ships onnxruntime (headers,
    # import libs and DLLs), built against each other: use both from there.
    set(PAROLI_DEPS_DIR "${CMAKE_SOURCE_DIR}/deps" CACHE PATH "Where piper-phonemize is downloaded")
    set(PIPER_PHONEMIZE_ROOT "${PAROLI_DEPS_DIR}/piper-phonemize" CACHE STRING "Path to piper-phonemize root directory")
    set(ORT_ROOT "${PIPER_PHONEMIZE_ROOT}" CACHE STRING "Path to ONNX Runtime root directory")
    if (NOT EXISTS "${PIPER_PHONEMIZE_ROOT}/lib/piper_phonemize.lib")
        # Same as tools/fetch-deps.ps1, done by CMake itself
        set(_pp_url "https://github.com/rhasspy/piper-phonemize/releases/download/2023.11.14-4/piper-phonemize_windows_amd64.zip")
        set(_pp_zip "${PAROLI_DEPS_DIR}/piper-phonemize_windows_amd64.zip")
        if (NOT EXISTS "${_pp_zip}")
            message(STATUS "Downloading ${_pp_url}")
            file(DOWNLOAD "${_pp_url}" "${_pp_zip}" SHOW_PROGRESS STATUS _pp_status)
            list(GET _pp_status 0 _pp_code)
            if (NOT _pp_code EQUAL 0)
                file(REMOVE "${_pp_zip}")
                message(FATAL_ERROR "Could not download piper-phonemize: ${_pp_status}")
            endif()
        endif()
        file(ARCHIVE_EXTRACT INPUT "${_pp_zip}" DESTINATION "${PAROLI_DEPS_DIR}")
        if (NOT EXISTS "${PIPER_PHONEMIZE_ROOT}/lib/piper_phonemize.lib")
            message(FATAL_ERROR "piper-phonemize not found in ${PIPER_PHONEMIZE_ROOT} after extracting ${_pp_zip}")
        endif()
    endif()
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS _SILENCE_ALL_CXX20_DEPRECATION_WARNINGS)
    add_compile_options(/utf-8 /EHsc /bigobj)
else()
    set(ORT_ROOT "" CACHE STRING "Path to ONNX Runtime root directory")
    set(PIPER_PHONEMIZE_ROOT "" CACHE STRING "Path to piper-phonemize root directory")
    if (ORT_ROOT STREQUAL "" OR PIPER_PHONEMIZE_ROOT STREQUAL "")
        message(FATAL_ERROR "ORT_ROOT and PIPER_PHONEMIZE_ROOT must be set")
    endif()
endif()

include_directories(${ORT_ROOT}/include ${PIPER_PHONEMIZE_ROOT}/include)
link_directories(${ORT_ROOT}/lib ${PIPER_PHONEMIZE_ROOT}/lib)

find_package(fmt CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(xtensor CONFIG REQUIRED)

add_library(piper STATIC piper/piper.cpp)
if (USE_RKNN)
    target_compile_definitions(piper PRIVATE USE_RKNN)
    target_sources(piper PRIVATE piper/rknn-inferer.cpp)
    target_link_libraries(piper PRIVATE rknnrt)
endif()
target_precompile_headers(piper PRIVATE piper/pch.hpp)
target_include_directories(piper PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/piper)
target_link_libraries(piper
    PUBLIC
    onnxruntime
    espeak-ng
    piper_phonemize
    fmt::fmt
    spdlog::spdlog
    nlohmann_json::nlohmann_json
    xtensor)
if (NOT WIN32)
    target_link_libraries(piper PUBLIC pthread dl)
endif()

add_executable(paroli-cli paroli-cli/main.cpp)
target_link_libraries(paroli-cli PRIVATE piper)

add_executable(paroli-stream paroli-stream/main.cpp)
target_link_libraries(paroli-stream PRIVATE piper)

set(PAROLI_TARGETS paroli-cli paroli-stream)

if (PAROLI_BUILD_SERVER)
    find_package(Drogon CONFIG REQUIRED)
    if (WIN32)
        find_package(Opus CONFIG REQUIRED)
        find_package(Ogg CONFIG REQUIRED)
        find_path(OPUSENC_INCLUDE_DIR opus/opusenc.h)
        find_library(OPUSENC_LIBRARY opusenc)
        find_package(soxr CONFIG QUIET)
        find_path(SOXR_INCLUDE_DIR soxr.h)
        find_library(SOXR_LIBRARY soxr)
        add_executable(paroli-server
            paroli-server/api.cpp
            paroli-server/OggOpusEncoder.cpp
            paroli-server/main.cpp)
        target_include_directories(paroli-server PRIVATE ${OPUSENC_INCLUDE_DIR} ${OPUSENC_INCLUDE_DIR}/opus ${SOXR_INCLUDE_DIR})
        target_link_libraries(paroli-server PRIVATE piper Drogon::Drogon Opus::opus Ogg::ogg ${OPUSENC_LIBRARY} ${SOXR_LIBRARY})
    else()
        find_package(Opus REQUIRED)
        add_executable(paroli-server
            paroli-server/api.cpp
            paroli-server/OggOpusEncoder.cpp
            paroli-server/main.cpp)
        target_link_libraries(paroli-server PRIVATE piper Drogon::Drogon soxr ${OPUS_LIBRARIES} opusenc ogg)
        target_include_directories(paroli-server PRIVATE ${OPUS_INCLUDE_DIRS})
    endif()
    target_precompile_headers(paroli-server PRIVATE paroli-server/pch.hpp)
    list(APPEND PAROLI_TARGETS paroli-server)
endif()

# Next to each executable: piper-phonemize/onnxruntime/espeak-ng DLLs and
# espeak-ng-data (paroli looks for it next to the .exe).
if (WIN32)
    foreach(t ${PAROLI_TARGETS})
        add_custom_command(TARGET ${t} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${PIPER_PHONEMIZE_ROOT}/bin/espeak-ng.dll
                ${PIPER_PHONEMIZE_ROOT}/bin/piper_phonemize.dll
                ${PIPER_PHONEMIZE_ROOT}/lib/onnxruntime.dll
                ${PIPER_PHONEMIZE_ROOT}/lib/onnxruntime_providers_shared.dll
                $<TARGET_FILE_DIR:${t}>
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${PIPER_PHONEMIZE_ROOT}/share/espeak-ng-data
                $<TARGET_FILE_DIR:${t}>/espeak-ng-data)
    endforeach()
endif()

install(TARGETS ${PAROLI_TARGETS} RUNTIME DESTINATION bin)
