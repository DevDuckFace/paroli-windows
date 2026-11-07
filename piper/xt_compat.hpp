#pragma once
// xtensor 0.26+ moved its headers into subfolders (containers/, views/,
// io/...). Include whichever layout is installed.
#if __has_include(<xtensor/containers/xarray.hpp>)
#include <xtensor/containers/xarray.hpp>
#include <xtensor/containers/xadapt.hpp>
#include <xtensor/io/xio.hpp>
#include <xtensor/views/xview.hpp>
#else
#include <xtensor/xarray.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#endif
