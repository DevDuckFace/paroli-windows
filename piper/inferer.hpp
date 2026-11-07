#pragma once

#include <string>
#include <vector>
#include <optional>

#include "xt_compat.hpp"

struct DecoderInferer {
  virtual ~DecoderInferer() = default;
  virtual std::vector<int16_t> infer(const xt::xarray<float>& z, const xt::xarray<float>& y_mask, const std::optional<xt::xarray<float>>& g) = 0;
  virtual void load(std::string modelPath, std::string accelerator) = 0;
};
