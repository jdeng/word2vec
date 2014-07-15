/*
Copyright (c) 2013, jackdeng@gmail.com
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Algorithm is based on matlab code from http://www.cs.toronto.edu/~hinton/MatlabForSciencePaper.html
/*
Training a deep autoencoder or a classifier 
on MNIST digits

Code provided by Ruslan Salakhutdinov and Geoff Hinton 
Permission is granted for anyone to copy, use, modify, or distribute this program and accompanying programs and documents for any purpose, provided this copyright notice is retained and prominently displayed, along with a note saying that the original programs are available from our web page. The programs and documents are distributed without any warranty, express or implied. As the programs were written for research purposes only, they have not been tested to the degree that would be advisable in any important application. All use of these programs is entirely at the user's own risk.
*/

// Conjugate Gradient implementation is based on matlab code at http://learning.eng.cam.ac.uk/carl/code/minimize/minimize.m
// % Copyright (C) 2001 - 2006 by Carl Edward Rasmussen (2006-09-08).

#pragma once

#include <numeric>
#include <vector>
#include <random>
#include <memory>
#include <fstream>
#include <chrono>

#include "v.h"

struct Batch: public std::pair<std::vector<Vector>::iterator, std::vector<Vector>::iterator>
{
  using Iterator = std::vector<Vector>::iterator;
  using Parent = std::pair<Iterator, Iterator>;
  template <typename... Arg> Batch(Arg&& ... arg): Parent(std::forward<Arg>(arg) ...) {}

  Iterator begin() const { return first; }
  Iterator end() const { return second; }
  size_t size() const { return std::distance(first, second); }
  bool empty() const  { return first == second; }

  Vector& operator[](size_t i) { return *(first + i); }
  const Vector& operator[](size_t i) const { return *(first + i); }
};

struct RBM
{
  enum class Type 
  {
    SIGMOID,
    LINEAR,
    EXP
  };

  Type type_ = Type::SIGMOID;
  Vector bias_visible_, bias_hidden_, bias_visible_inc_, bias_hidden_inc_;
  Vector weight_, weight_inc_;

  struct Conf 
  {
    float momentum_ = 0.5, weight_cost_ = 0.0002, learning_rate_ = 0.1;
  };

  static float sigmoid(float x) { return (1.0 / (1.0 + exp(-x))); }

  static const Vector& bernoulli(const Vector& input, Vector& output)
  { 
    static std::default_random_engine eng(::time(NULL));
    static std::uniform_real_distribution<float> rng(0.0, 1.0);

    for (size_t i=0; i<input.size(); ++i) { output[i] = (rng(eng) < input[i])? 1.0 : 0.0; } 
    return output;
  }

  static const Vector& add_noise(const Vector& input, Vector& output)
  { 
    static std::default_random_engine eng(::time(NULL));
    static std::normal_distribution<float> rng(0.0, 1.0);

    for (size_t i=0; i<input.size(); ++i) { output[i] = input[i] + rng(eng); }
    return output;
  }

  RBM() {}

  RBM(size_t visible, size_t hidden)
    : bias_visible_(visible), bias_hidden_(hidden), weight_(visible * hidden)
    , bias_visible_inc_(visible), bias_hidden_inc_(hidden), weight_inc_(visible * hidden)
  {
    static std::default_random_engine eng(::time(NULL));
    static std::normal_distribution<float> rng(0.0, 1.0);
    for (auto& x: weight_) x = rng(eng) * .1;
  }

  size_t num_hidden() const { return bias_hidden_.size(); }
  size_t num_visible() const { return bias_visible_.size(); }
  size_t num_weight() const { return weight_.size(); }

  int mirror(const RBM& rbm)
  {
    size_t n_visible = bias_visible_.size(), n_hidden = bias_hidden_.size();
    if (n_hidden != rbm.num_visible() || n_visible != rbm.num_hidden()) { 
      std::cout << "not mirrorable" << std::endl;
      return -1;
    }
    
    bias_visible_ = rbm.bias_hidden_;
    bias_hidden_ = rbm.bias_visible_;
    for (size_t i = 0; i < n_visible; ++i) {
       for (size_t j = 0; j < n_hidden; ++j) {
        weight_[j * n_visible + i] = rbm.weight_[i * n_hidden + j];
      }
    }
    return 0;  
  }

  const Vector& activate_hidden(const Vector& visible, Vector& hidden) const {
    return RBM::activate_hidden(visible, hidden, bias_hidden_, weight_, type_);
  }

  template <class Vector1, class Vector2, class Vector3>
  static const Vector2& activate_hidden(const Vector1& visible, Vector2& hidden, const Vector3& bias_hidden, const Vector3& weight, Type type)
  {
    size_t n_visible = visible.size(), n_hidden = hidden.size();

    std::fill(hidden.begin(), hidden.end(), 0);
    for (size_t i = 0; i < n_hidden; ++i) {
      float *xd = const_cast<float *>(weight.data() + i * n_visible);
      float s = v::dot(visible, v::LightVector(xd, xd + n_visible));
      s += bias_hidden[i];

      if (type == Type::SIGMOID) s = sigmoid(s);
      else if (type == Type::EXP) s = exp(s);

      hidden[i] = s;
    }

    return hidden;
  }

  const Vector& activate_visible(const Vector& hidden, Vector& visible) const
  {
    size_t n_visible = bias_visible_.size(), n_hidden = bias_hidden_.size();

    std::fill(visible.begin(), visible.end(), 0);
    for (size_t i = 0; i < n_visible; ++i) {
      float s = 0;
      for (size_t j = 0; j < n_hidden; ++j) s += hidden[j] * weight_[j * n_visible+ i];
      s += bias_visible_[i];

      s = sigmoid(s);
      visible[i] = s;
    }

    return visible;
  }

  float train(Batch inputs, const Conf& conf)
  {
    size_t n_visible = bias_visible_.size(), n_hidden = bias_hidden_.size();
    float momentum = conf.momentum_, learning_rate = conf.learning_rate_, weight_cost = conf.weight_cost_;

    // temporary results
    Vector v1(n_visible), h1(n_hidden), v2(n_visible), h2(n_hidden), hs(n_hidden);

    //delta
    Vector gw(n_visible * n_hidden), gv(n_visible), gh(n_hidden);
    for (auto const& input: inputs) {
      v1 = input;
      this->activate_hidden(v1, h1);
      this->activate_visible((type_ == Type::LINEAR? add_noise(h1, hs): bernoulli(h1, hs)), v2);
      this->activate_hidden(v2, h2);

      for (size_t i = 0; i < n_visible; ++i) {
        for (size_t j = 0; j < n_hidden; ++j) gw[j * n_visible + i] += h1[j] * v1[i] - h2[j] * v2[i];
      }

//      gh += (h1 - h2);
//      gv += (v1 - v2);
      v::saxpy2(gh, 1.0, h1, -1.0, h2);
      v::saxpy2(gv, 1.0, v1, -1.0, v2);
    }

    //update
    size_t n_samples = inputs.size();
//    gw /= float(n_samples);
//    gw -= weight_ * weight_cost;
    v::saxpy(1.0/n_samples, gw, -weight_cost, weight_);
//    weight_inc_ = weight_inc_ * momentum + gw * learning_rate;
    v::saxpy(momentum, weight_inc_, learning_rate, gw);

//    weight_ += weight_inc_;
    v::saxpy(weight_, 1.0, weight_inc_);

//    gh /= float(n_samples); 
//    bias_hidden_inc_ = bias_hidden_inc_ * momentum + gh * learning_rate;
    v::saxpy(momentum, bias_hidden_inc_, learning_rate / n_samples, gh);
//    bias_hidden_ += bias_hidden_inc_;
    v::saxpy(bias_hidden_, 1.0, bias_hidden_inc_);

//    gv /= float(n_samples); 
//    bias_visible_inc_ = bias_visible_inc_ * momentum + gv * learning_rate;
    v::saxpy(momentum, bias_visible_inc_, learning_rate / n_samples, gv);
//    bias_visible_ += bias_visible_inc_;
    v::saxpy(bias_visible_, 1.0, bias_visible_inc_);

//    float error = sqrt(gv.dot(gv) / n_visible);
    v::scale(gv, 1.0/n_samples);
    float error = sqrt(v::dot(gv, gv) / n_visible);
//    std::cout << "error: " << error << ", energy: " << free_energy() << std::endl;

    return error;
  }

  virtual float free_energy() const
  {
    size_t n_visible = bias_visible_.size(), n_hidden = bias_hidden_.size();
    float s = 0;
    for (size_t i = 0; i < n_visible; ++i) {
      for (size_t j = 0; j < n_hidden; ++j) 
        s += weight_[j * n_visible+ i] * bias_hidden_[j] * bias_visible_[i];
    }
    return -s;
  }

  template <typename T> static void _write(std::ostream& os, const T& v) { os.write(reinterpret_cast<const char *>(&v), sizeof(v)); }
  void store(std::ostream& os) const
  {
    int type = (int)type_;
    size_t n_visible = bias_visible_.size();
    size_t n_hidden = bias_hidden_.size();

    _write(os, type); _write(os, n_visible); _write(os, n_hidden);
    for (float v: bias_visible_) _write(os, v);
    for (float v: bias_hidden_) _write(os, v);
    for (float v: weight_) _write(os, v);
  }

  template <typename T> static void _read(std::istream& is, T& v) { is.read(reinterpret_cast<char *>(&v), sizeof(v)); }
  void load(std::istream& is)
  {
    int type = 0;
    size_t n_visible = 0, n_hidden = 0;
    _read(is, type); _read(is, n_visible); _read(is, n_hidden);

    type_ = (Type)type;
    bias_visible_.resize(n_visible);
    bias_hidden_.resize(n_hidden);
    weight_.resize(n_visible * n_hidden);

    for (float& v: bias_visible_) _read(is, v);
    for (float& v: bias_hidden_) _read(is, v);
    for (float& v: weight_) _read(is, v);
  }
};

using RBMP = std::unique_ptr<RBM>;
struct LRBM // layered RBM
{
  struct Conf
  {
    int max_epoch_ = 20, max_batches_ = 1000, batch_size_ = 30;
  };
  
  std::vector<RBMP> rbms_;

  RBMP& output_layer() { return rbms_[rbms_.size() - 1]; }
  size_t max_layer() const { return rbms_.size(); }

  int build(const std::vector<int>& layers, const std::vector<int>& adjust = std::vector<int>())
  {
    if (layers.size() <= 1) return -1;

    for (size_t i=0; i<layers.size() - 1; ++i) {
      int n_visible= layers[i] + (adjust.empty()? 0: adjust[i]);
      int n_hidden = layers[i+1];
      
      std::cout << "New RBM " << n_visible << " -> " << n_hidden << std::endl;
      rbms_.push_back(std::unique_ptr<RBM>(new RBM(n_visible, n_hidden)));
    }

    return 0;
  }

  std::vector<int> offsets(int start) const
  {
    int n_layers = rbms_.size() - start;
    std::vector<int> dims(n_layers + 1);
    dims[0] = 0;
    int total = 0;
    for(size_t i=0; i<n_layers; ++i) {
      total += (rbms_[i + start]->num_visible() + 1) * rbms_[i + start]->num_hidden();
      dims[i+1] = total;
    }
    return dims;
  }

  void to_image(Vector& image, int& width, int& height)
  {
    width = 0; height = 0;
    auto& rbms = this->rbms_;
    for (auto& rbm: rbms) {
      if (width < rbm->num_hidden() + 1) width = rbm->num_hidden() + 1;
      height += (rbm->num_visible() + 2);  
    }
    image.resize(width * height);

    size_t y_offset = 0;
    for (auto& rbm: rbms) {
      size_t n_visible = rbm->num_visible();
      size_t n_hidden = rbm->num_hidden();
      size_t x_offset = (width - n_hidden) / 2;
  
      for (size_t j=0; j<n_hidden; ++j)
          image[y_offset * width + x_offset + j] = rbm->bias_hidden_[j];
      for (size_t i=0; i<n_visible; ++i) {
        for (size_t j=0; j<n_hidden; ++j)
//          image[(y_offset + i) * width + x_offset + j] = rbm->weight_[i * n_hidden + j];
          image[(y_offset + i) * width + x_offset + j] = rbm->weight_[j * n_visible+ i];
        image[(y_offset + i) * width + x_offset + n_hidden] = rbm->bias_visible_[i];
      }
      y_offset += n_visible + 2;
    }
  }

  void store(std::ostream& os) const
  {
    int32_t count = rbms_.size();
    os.write(reinterpret_cast<char *>(&count), sizeof(count));
    for (auto const& rbm: rbms_) rbm->store(os);
  }

  void load(std::istream& is)
  {
    int32_t count = 0;
    is.read(reinterpret_cast<char *>(&count), sizeof(count));

    rbms_.clear();
    for (size_t i = 0; i < count; ++i) 
    {
      RBMP rbm(new RBM());
      rbm->load(is);
      rbms_.push_back(std::move(rbm));
    }
  }
};

struct DeepBeliefNet : public LRBM
{
  float free_energy() const {
    return std::accumulate(rbms_.begin(), rbms_.end(), 0.0f, [](float x, const RBMP& rbm) { return x + rbm->free_energy(); });
  }

  int train(std::vector<Vector>& inputs, std::vector<Vector>& labels, int max_layer, LRBM::Conf& conf, std::function<void(DeepBeliefNet&)> progress = nullptr)
  {
    int n_samples = inputs.size(), n_labels = labels.size();
    if (n_labels > 0 && n_samples != n_labels) {
      std::cerr << "# inputs does not match # labels" << std::endl;
      return -1;
    }

    int max_epoch = conf.max_epoch_, batch_size = conf.batch_size_, max_batches = std::min(conf.max_batches_, n_samples / batch_size); 
    std::vector<Vector> probs(n_samples);
    for(int layer = 0; layer < max_layer; ++layer) {
      auto& rbm = this->rbms_[layer];
      RBM::Conf conf;
      //XXX: more epochs and lower learning rate for linear rbm
      if (rbm->type_ == RBM::Type::LINEAR) { max_epoch = 100; conf.learning_rate_ = 0.001; }

      for (int epoch = 0; epoch < max_epoch; ++epoch) {
        if (progress) { progress(*this); }

        //XXX: update momentum
        if (epoch > 5) conf.momentum_ = .9f;

        for (size_t batch = 0; batch < max_batches; ++batch) {
          int start = batch * batch_size, end = std::min(start + batch_size, n_samples);

          Batch data;
          if (layer == 0) 
            data = Batch{inputs.begin() + start, inputs.begin() + end};
          else 
            data = Batch{probs.begin() + start, probs.begin() + end};
        
          float error = rbm->train(data, conf);

          if ((batch + 1) % 10 == 0) {
            std::cout << "layer: " << layer << ", epoch: " << epoch << ", batch: " << batch + 1 << ", error: " << error << ", energy: " << this->free_energy() << std::endl;
          }

          //save outputs to probs at last epoch
          if (epoch == max_epoch - 1) {
            auto it = data.begin();
            for(int i = start; i < end; ++i) {
              Vector output(rbm->num_hidden());
              rbm->activate_hidden(*it++, output);
              output.swap(probs[i]);
            }

            //attach labels for last layer
            if (layer > 0 && layer + 1 == max_layer - 1 && !labels.empty()) {
              size_t input_size = probs[start].size(), label_size = labels.front().size();
              for (size_t i = start; i < end; ++i) {
                const Vector& label = labels[i];
                Vector& input = probs[i];
                input.resize(input_size + label_size);
                std::copy(label.begin(), label.end(), input.begin() + input_size);
              }
            }
          }
        }  
      }
    }
  
    if (progress) { progress(*this); }
    return 0;
  }

  int predict(const Vector& sample, Vector& output, Vector& probs)
  {
    static std::default_random_engine eng(::time(NULL));
    std::uniform_real_distribution<float> rng(0.0, 1.0);
  
    Vector input(sample);
    int n_layers = rbms_.size();
    for (int i =0; i<n_layers - 1; ++i) {
      const RBMP& rbm = rbms_[i];
      size_t n_visible = rbm->num_visible(), n_hidden = rbm->num_hidden();
      
      Vector next(n_hidden);
      rbm->activate_hidden(input, next); 
      input.swap(next);  
    }

    RBMP& rbm = rbms_[n_layers - 1];
    size_t n_visible = rbm->num_visible();
    size_t n_hidden = rbm->num_hidden();
    size_t n_input = input.size();
    if (n_input  + output.size() != n_visible) {
      return -1;
    }

    // attach zero-ed labels
    if (n_visible > n_input) input.resize(n_visible);

    Vector h1(n_hidden);
    rbm->activate_hidden(input, h1);

    if (! probs.empty()) 
      probs = h1;

    if (! output.empty()) {
      Vector hs(n_hidden), v2(n_visible);
      rbm->activate_visible(RBM::bernoulli(h1, hs), v2);
      std::copy(v2.begin() + n_input, v2.end(), output.begin());
    }

    return 0;    
  }

  struct GradientContext
  {
    int max_iteration_;

    int epoch_;
    Batch inputs_;
    Batch targets_;
    int start_layer_;

    std::vector<std::vector<Vector>>& probs_;

    GradientContext(Batch inputs, std::vector<std::vector<Vector>> & probs, int epoch)
      : max_iteration_(3), epoch_(epoch), inputs_(inputs), start_layer_(0), probs_(probs)
    {}
  };

  int gradient(GradientContext& ctx, const Vector& weights, Vector& weight_incs, float& cost)
  {
    Batch& inputs = ctx.inputs_;
    std::vector<std::vector<Vector>>& probs = ctx.probs_; 
    bool has_targets = !ctx.targets_.empty();

    int max_layer = this->rbms_.size();

    size_t n_hidden = rbms_.back()->num_hidden();
    size_t n_samples = inputs.size();
    std::vector<Vector> diffs(n_samples);

    auto cstart = std::chrono::high_resolution_clock::now();
    auto dims = this->offsets(ctx.start_layer_);

    std::fill(weight_incs.begin(), weight_incs.end(), 0);
    cost = 0;
    float error = 0;
    for (size_t sample = 0; sample < n_samples; ++sample) { 
      const Vector& input = inputs[sample];
      v::LightVector bias_hidden, weight;
      for (int layer=0; layer < max_layer; ++layer) {
        const RBMP& rbm = this->rbms_[layer];
        if (layer < ctx.start_layer_ || weights.empty()) { 
          float *start = const_cast<float *>(rbm->weight_.data()), *end = start + rbm->num_weight();
          weight = v::LightVector(start, end);
          start = const_cast<float *>(rbm->bias_hidden_.data()); end = start + rbm->num_hidden();
          bias_hidden = v::LightVector(start, end);
        }
        else { 
          float *start = const_cast<float *>(weights.data()) + dims[layer - ctx.start_layer_], *end = start + rbm->num_weight();
          weight = v::LightVector(start, end);
          start = end; end = start + rbm->num_hidden();
          bias_hidden = v::LightVector(start, end);
        }

        Vector& output = probs[layer][sample];
        const Vector& _input = (layer == 0? input: probs[layer - 1][sample]);
        RBM::activate_hidden(_input, output, bias_hidden, weight, rbm->type_);
      }

      //output
      Vector& result = probs[max_layer - 1][sample]; 
      Vector& diff = diffs[sample];
      diff.resize(n_hidden);

      if (has_targets) {
        float s = std::accumulate(result.begin(), result.end(), 0.0);
        v::scale(result, 1.0/s);

        const Vector& target = ctx.targets_[sample];
        for(size_t i=0; i<n_hidden; ++i) {
          diff[i] = (result[i] - target[i]);
          cost += target[i] * log(result[i]); 
          error += diff[i] * diff[i];
        }
      } else {
        for(size_t i=0; i<n_hidden; ++i) {
          diff[i] = (result[i] - input[i]) / n_samples;
          cost += input[i] * log(result[i]) + (1 - input[i]) * log(1 - result[i]);  
          error += (result[i] - input[i]) * (result[i] - input[i]);
        }
      }
    }

    cost = -cost;
    if (! has_targets) cost *= 1/ n_samples;

    //calculate gradient
    for (int layer=max_layer - 1; layer >= 0; --layer) {
      if (layer < ctx.start_layer_) 
        break;

      if (layer != max_layer - 1) {
        const RBMP& rbm = this->rbms_[layer + 1];
//        const Vector& weight = rbm->weight_; 
        size_t n_visible = rbm->num_visible(), n_hidden = rbm->num_hidden();
        size_t offset = dims[layer + 1 - ctx.start_layer_];
        v::LightVector weight(const_cast<float *>(weights.data()) + offset, const_cast<float *>(weights.data()) + offset + rbm->num_weight());
        for (size_t sample = 0; sample < n_samples; ++sample) { 
          Vector diff(n_visible);
          for (size_t j=0; j<n_visible; ++j) {
            float s = 0;
            for (size_t k=0; k<n_hidden; ++k) {
//              s += diffs[sample][k] * weight[j * n_hidden + k];
              s += diffs[sample][k] * weight[k * n_visible + j];
            }
            if (rbms_[layer]->type_ != RBM::Type::LINEAR)
              s *= probs[layer][sample][j] * (1.0 - probs[layer][sample][j]);
            diff[j] = s;
          }
          diffs[sample].swap(diff);
        } 
      }

      RBMP& rbm = this->rbms_[layer];
      size_t n_visible = rbm->num_visible(), n_hidden = rbm->num_hidden();
      size_t offset = dims[layer - ctx.start_layer_];

      for (size_t sample = 0; sample < n_samples; ++sample) { 
        const auto& v = (layer > 0? probs[layer-1][sample] : inputs[sample]);
        const auto& d = diffs[sample];
        for (size_t j=0; j<n_visible; ++j) {
          for (size_t k=0; k<n_hidden; ++k) {
            weight_incs[offset + k * n_visible + j] += v[j] * d[k];
          }
        }
        for (size_t k=0; k<n_hidden; ++k) {
            weight_incs[offset + n_visible * n_hidden + k] += d[k];
        }
      }
    }

    auto cend = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(cend - cstart).count() / 1000.0;
    std::cout << "evaluating: cost=" << cost << ", error=" << error / n_samples << " in " << duration << "ms" << std::endl;

    return 0;
  }

  // translate into C++ from matlab code
  //    http://learning.eng.cam.ac.uk/carl/code/minimize/minimize.m
  int minimize(GradientContext& ctx)
  {
    const float INT = 0.1, EXT = 3.0;
    const float SIG = 0.1, RHO = SIG / 2.0, RATIO = 10;
    const int max_iteration = ctx.max_iteration_;

    // initialize
    float cost = 0;
    auto dims = this->offsets(ctx.start_layer_); 
    Vector weights(dims.back()), weight_incs(dims.back()); 

    {
      auto offset = weights.begin();
      for (size_t i=ctx.start_layer_; i<this->rbms_.size(); ++i) {
        const RBMP& rbm = this->rbms_[i];
        std::copy(rbm->weight_.begin(), rbm->weight_.end(), offset);
        offset += rbm->num_weight();
        std::copy(rbm->bias_hidden_.begin(), rbm->bias_hidden_.end(), offset);
        offset += rbm->num_hidden();
      }
    }

    this->gradient(ctx, weights, weight_incs, cost);
    
    Vector df0(weight_incs);
    Vector s(df0); v::scale(s, -1.0);
    float d0 = -v::dot(s, s), f0 = cost;
    float d3 = 0, x3 = 1.0 / (1 - d0);

//    std::cout << "d3=" << d3 << ", d0=" << d0 << ",f0=" << f0 << std::endl;

    bool failed = false;
    // line search
    for (int i=0; i<max_iteration; ++i) {
      // extrapolation
      float best_cost = f0;
      Vector best_weights(weights), best_weight_incs(weight_incs);

      float f3 = 0;
      Vector df3(weights.size());

      int M = 20;
      float f1 = 0, x1 = 0, d1 = 0;
      float f2 = 0, x2 = 0, d2 = 0;
      while (true) {
        x2 = 0; f2 = f0; d2 = d0; 
        f3 = f0; df3 = df0;

        while (true) {
          if (M -- < 0) break;
          
          Vector tmp_weights(weights);
//          tmp_weights += s * x3;
          v::saxpy(tmp_weights, x3, s);
          this->gradient(ctx, tmp_weights, weight_incs, cost);
          f3 = cost; df3 = weight_incs;
          if (std::isfinite(cost) && v::isfinite(weight_incs)) {
            //found one and save best result if available
            if (f3 < best_cost) {
              best_cost = f3;
              best_weights = tmp_weights;
              best_weight_incs = weight_incs;
            }
            break;
          }

          //back off and retry
          x3 = (x2 + x3) / 2.0;
        }

        // check slope and done extrapolation?
        d3 = v::dot(df3,s);
        if (d3 > SIG*d0 || f3 > f0 + x3*RHO*d0 || M <= 0) break;

        x1 = x2; f1 = f2; d1 = d2;
        x2 = x3; f2 = f3; d2 = d3;  

        // cubic extrapolation
        float dx = x2-x1;
        float A = 6.0*(f1-f2) + 3.0*(d2+d1)*dx;
        float B = 3.0*(f2-f1) - (2.0*d1+d2)*dx;
        x3 = x1-d1*dx*dx/(B+sqrt(B*B-A*d1*dx));
  
        // keep it in range
        float upper = x2 * EXT, lower = x2 + INT * dx;
        if (!std::isfinite(x3) || x3 < 0 || x3 > upper) x3 = upper;
        else if (x3 < lower) x3 = lower;
      }

      // interpolation
      float f4 = 0, x4 = 0, d4 = 0;
      while ((std::abs(d3) > -SIG*d0 || f3 > f0 + x3*RHO*d0) && M > 0) {
        if (d3 > 0 || f3 > f0+x3*RHO*d0) {
          x4 = x3; f4 = f3; d4 = d3;        
        } else {
          x2 = x3; f2 = f3; d2 = d3;
        }

        float dx = x4 - x2;
        if (f4 > f0) {
          x3 = x2-(0.5*d2*dx*dx)/(f4-f2-d2*dx);  // quadratic interpolation
        } else {
          float A = 6*(f2-f4)/dx+3*(d4+d2);     // cubic interpolation
          float B = 3*(f4-f2)-(2*d2+d4)*dx;
          x3 = x2+(sqrt(B*B-A*d2*dx*dx)-B)/A; 
        }

        if (! std::isfinite(x3)) {
//          std::cout << "x3 = " << x3 << " not usable" << std::endl;
          x3 = (x2 + x4) / 2;
        }
        
        // keep it in range
        x3 = std::max(std::min(x3, x4-INT*(x4-x2)),x2+INT*(x4-x2));

        Vector tmp_weights(weights);
//        tmp_weights += s * x3;
        v::saxpy(tmp_weights, x3, s);
        this->gradient(ctx, tmp_weights, weight_incs, cost);
        f3 = cost; df3 = weight_incs;
        if (f3 < best_cost) {
          best_cost = f3;
          best_weights = tmp_weights;
          best_weight_incs = weight_incs;
        }

        --M;
//        d3 = df3.dot(s);
        d3 = v::dot(df3,s);
      }

      if (std::abs(d3) < -SIG*d0 && f3 < f0 + x3*RHO*d0) { // succeeded
//        weights += s * x3; 
        v::saxpy(weights, x3, s);
        f0 = f3; 
//        s *= (df3.dot(df3) - df3.dot(df0)) / df0.dot(df0); s -= df3; // Polack-Ribiere CG direction
        float g = (v::dot(df3,df3) - v::dot(df3, df0)) / v::dot(df0, df0);
        v::saxpy(g, s, -1.0, df3); // Polack-Ribiere CG direction
//        d3 = d0; d0  = df3.dot(s); df0 = df3; 
        d3 = d0; d0  = v::dot(df3, s); df0 = df3; 
        if (d0 > 0) {
//          s = -df0; d0 = -df0.dot(df0);
          s = df0; v::scale(s, -1.0); d0 = -v::dot(df0, df0);
        }

        x3 = x3 * std::min(RATIO, float(d3 / (d0 - 1e-37)));
        failed = false;
        std::cout << "found: iteration i=" << i << ", cost=" << f3 << std::endl;
      } else { // failed
        std::cout << "x3 = " << x3 << " failed" << std::endl;
        weights = best_weights; f0 = best_cost; df0 = best_weight_incs; 
        if (failed) break;  

//        s = -df0; d0 = - s.dot(s); x3 = 1.0/(1.0-d0);
        s = df0; v::scale(s, -1.0); d0 = -v::dot(s, s); x3 = 1.0/(1.0-d0);
        failed = true;
      }
    }

    //apply the new weights
    {
      auto offset = weights.begin();
      for (size_t i=ctx.start_layer_; i<this->rbms_.size(); ++i) {
        const RBMP& rbm = this->rbms_[i];
        std::copy(offset, offset + rbm->num_weight(), rbm->weight_.begin());
        offset += rbm->num_weight();
        std::copy(offset, offset + rbm->num_hidden(), rbm->bias_hidden_.begin());
        offset += rbm->num_hidden();
      }
    }

    std::cout << "applying new weights to " << ctx.start_layer_ << "+" << std::endl;
    return 0;
  }

  virtual int pretrain(std::vector<Vector>& inputs, LRBM::Conf& conf, std::function<void(DeepBeliefNet&)> progress = nullptr)
  {
    static std::vector<Vector> nil;
    return train(inputs, nil, this->rbms_.size() - 1, conf, progress);
  }

  int backprop(std::vector<Vector>& inputs, std::vector<Vector>& targets, LRBM::Conf& conf, std::function<void(DeepBeliefNet&)> progress = nullptr) 
  {
    int batch_size = conf.batch_size_, max_epoch = conf.max_epoch_, max_batches = conf.max_batches_; 
    int max_layer = this->rbms_.size();
    
    std::vector<std::vector<Vector>> probs(max_layer);
    for (int i = 0; i < max_layer; ++i) {
      const RBMP& rbm = this->rbms_[i];
      probs[i].resize(batch_size);
      for (auto &v: probs[i]) { v.resize(rbm->num_hidden()); }
    }

    for (int epoch = 0; epoch < max_epoch; ++epoch) {
      for (int j = 0; j < max_batches; ++j) {
        int start = j * batch_size, end = start + std::min(batch_size, int(inputs.size()) - j * batch_size);
        std::cout << "epoch: " << epoch << ", batch: " << j << ", samples: "<< (end - start) << std::endl;
        if (progress) progress(*this);
        GradientContext ctx(Batch(inputs.begin() + start, inputs.begin() + end), probs, epoch);
//        ctx.start_layer_ = (epoch > std::min(6, max_epoch / 2)? 0: this->rbms_.size() - 1);
        if (! targets.empty())
          ctx.targets_ = Batch(targets.begin() + start, targets.begin() + end);
        this->minimize(ctx);
      }
    }

    if (progress) progress(*this);
    return 0;
  }
};

