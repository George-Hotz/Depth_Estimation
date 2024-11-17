#include "preprocess.h"

static uint8_t *img_buffer_device = nullptr;

__global__ void warpaffine_kernel(
  uint8_t *src, int src_line_size, int src_width,
  int src_height, float *dst, int dst_width,
  int dst_height, uint8_t const_value_st,
  AffineMatrix d2s, int edge)
{
  int position = blockDim.x * blockIdx.x + threadIdx.x;
  if (position >= edge)
    return;

  float mean[] {0.485, 0.456, 0.406};
  float std[] {0.229, 0.224, 0.225};

  // 从d2s中读取变换矩阵
  float m_x1 = d2s.value[0];
  float m_y1 = d2s.value[1];
  float m_z1 = d2s.value[2];
  float m_x2 = d2s.value[3];
  float m_y2 = d2s.value[4];
  float m_z2 = d2s.value[5];

  int dx = position % dst_width; // 计算当前线程对应的目标图像的x坐标
  int dy = position / dst_width; // 计算当前线程对应的目标图像的y坐标

  float src_x = m_x1 * dx + m_y1 * dy + m_z1 + 0.5f;
  float src_y = m_x2 * dx + m_y2 * dy + m_z2 + 0.5f;
  float c0, c1, c2;

  if (src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height)
  {
    // 超出边界的像素点用const_value_st填充
    c0 = const_value_st;
    c1 = const_value_st;
    c2 = const_value_st;
  }else{
    // 双线性插值，实现图像的放大缩小
    int y_low = floorf(src_y);
    int x_low = floorf(src_x);
    int y_high = y_low + 1;
    int x_high = x_low + 1;

    uint8_t const_value[] = {const_value_st, const_value_st, const_value_st};
    float ly = src_y - y_low;
    float lx = src_x - x_low;
    float hy = 1 - ly;
    float hx = 1 - lx;
    float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
    uint8_t *v1 = const_value;
    uint8_t *v2 = const_value;
    uint8_t *v3 = const_value;
    uint8_t *v4 = const_value;

    if (y_low >= 0)
    {
      if (x_low >= 0)
        v1 = src + y_low * src_line_size + x_low * 3;

      if (x_high < src_width)
        v2 = src + y_low * src_line_size + x_high * 3;
    }

    if (y_high < src_height)
    {
      if (x_low >= 0)
        v3 = src + y_high * src_line_size + x_low * 3;

      if (x_high < src_width)
        v4 = src + y_high * src_line_size + x_high * 3;
    }

    c0 = w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0];
    c1 = w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1];
    c2 = w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2];
  }

  // bgr to rgb
  float t = c2;
  c2 = c0;
  c0 = t;

  // normalization
  c0 = c0 / 255.0f;
  c1 = c1 / 255.0f;
  c2 = c2 / 255.0f;

  // imagenet normalization
  // c0 = (c0-mean[0])/std[0];
  // c1 = (c1-mean[1])/std[1];
  // c2 = (c2-mean[2])/std[2];

  // rgbrgbrgb to rrrgggbbb
  int area = dst_width * dst_height;
  float *pdst_c0 = dst + dy * dst_width + dx;
  float *pdst_c1 = pdst_c0 + area;
  float *pdst_c2 = pdst_c1 + area;
  *pdst_c0 = c0;
  *pdst_c1 = c1;
  *pdst_c2 = c2;
}


void cuda_preprocess(
    uint8_t *src, int src_width, int src_height,
    float *dst, int dst_width, int dst_height)
{

  int img_size = src_width * src_height * 3;
  CUDA_CHECK(cudaMemcpy(img_buffer_device, src, img_size, cudaMemcpyHostToDevice));

  // 计算变换矩阵
  AffineMatrix s2d, d2s;
  float scale = std::min(dst_height / (float)src_height, dst_width / (float)src_width);

  s2d.value[0] = scale;
  s2d.value[1] = 0;
  s2d.value[2] = -scale * src_width * 0.5 + dst_width * 0.5;
  s2d.value[3] = 0;
  s2d.value[4] = scale;
  s2d.value[5] = -scale * src_height * 0.5 + dst_height * 0.5;

  cv::Mat m2x3_s2d(2, 3, CV_32F, s2d.value);
  cv::Mat m2x3_d2s(2, 3, CV_32F, d2s.value);
  cv::invertAffineTransform(m2x3_s2d, m2x3_d2s);

  memcpy(d2s.value, m2x3_d2s.ptr<float>(0), sizeof(d2s.value));

  // 一个线程处理一个像素点，一共需要 dst_height * dst_width 个线程
  int jobs = dst_height * dst_width;
  int threads = 256;
  int blocks = ceil(jobs / (float)threads);
  // 调用kernel函数

  warpaffine_kernel<<<blocks, threads>>>(
      img_buffer_device, src_width * 3, src_width,
      src_height, dst, dst_width,
      dst_height, 0, d2s, jobs);
}

void cuda_batch_preprocess(std::vector<cv::Mat> &img_batch,
                           float *dst, int dst_width, int dst_height)
{
  int dst_size = dst_width * dst_height * 3;
  for (size_t i = 0; i < img_batch.size(); i++)
  {
    cuda_preprocess(img_batch[i].ptr(), img_batch[i].cols, img_batch[i].rows, &dst[dst_size * i], dst_width, dst_height);
  }
}

void cuda_preprocess_init(int max_image_size)
{
  // prepare input data in device memory
  CUDA_CHECK(cudaMalloc((void **)&img_buffer_device, max_image_size * 3));
}

void cuda_preprocess_destroy()
{
  CUDA_CHECK(cudaFree(img_buffer_device));
}

// 使用cuda预处理所有步骤
void Preprocess_gpu(const cv::Mat &src, int inputW, int inputH, float *input_device_buffer)
{
  cuda_preprocess((uint8_t *)src.ptr(), src.cols, src.rows, input_device_buffer, inputW, inputH);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void Preprocess_cpu(const cv::Mat &img, int inputW, int inputH, void *input_host_buffer)
{
  cv::Mat resized;
  cv::resize(img, resized, cv::Size(inputW, inputH));
  cv::Mat rgb;
  cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
  cv::Mat normalized;
  rgb.convertTo(normalized, CV_32FC3);
  cv::subtract(normalized, cv::Scalar(127.5, 127.5, 127.5), normalized);
  cv::divide(normalized, cv::Scalar(127.5, 127.5, 127.5), normalized);
  // split it into three channels
  std::vector<cv::Mat> nchw_channels;
  cv::split(normalized, nchw_channels);

  for (auto &img : nchw_channels)
  {
      img = img.reshape(1, 1);
  }

  cv::Mat nchw;
  cv::hconcat(nchw_channels, nchw);

  memcpy(input_host_buffer, nchw.data, 3 * inputH * inputW * sizeof(float));
}