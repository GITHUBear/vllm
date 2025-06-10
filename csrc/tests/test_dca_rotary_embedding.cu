#include <torch/torch.h>
#include <vector>

using namespace torch::indexing;

extern void dca_rotary_embedding(
  torch::Tensor& positions,
  torch::Tensor& query,
  torch::Tensor& key,
  int64_t head_size,
  torch::Tensor& cos_sin_q_cache,
  torch::Tensor& cos_sin_qc_cache,
  torch::Tensor& cos_sin_qc_no_clamp_cache,
  torch::Tensor& cos_sin_q_inter_cache,
  torch::Tensor& out,
  int64_t chunk_len,
  bool is_neox
);

int main() {
  // int64_t num_head = 1;
  int64_t head_size = 128;
  int64_t rotary_dim = 128;
  int64_t chunk_size = 100;
  int64_t local_size = 20;
  int64_t chunk_len = chunk_size - local_size;
  bool is_neox = true;

  torch::Device device(torch::kCUDA);
  torch::Device cpu_device(torch::kCPU);

  std::vector<int64_t> seq_lens = {20, 10, 30};
  // Step 1: 生成 positions
  std::vector<torch::Tensor> position_tensors;
  for (int64_t n : seq_lens) {
    // 生成 [0, 1, ..., n-1] 的张量，类型为 int64，设备为 CUDA
    torch::Tensor pos = torch::arange(0, n, torch::TensorOptions().device(device).dtype(torch::kInt64));
    position_tensors.push_back(pos);
  }
  torch::Tensor positions = torch::cat(position_tensors, 0);

  int64_t total_length = std::accumulate(seq_lens.begin(), seq_lens.end(), 0LL); // 计算总长度
  torch::Tensor query = torch::rand({total_length, head_size}, torch::TensorOptions().device(cpu_device).dtype(torch::kFloat32));
  torch::Tensor key = torch::rand_like(query);

  torch::Tensor cos_sin_q_cache = torch::rand({chunk_len, rotary_dim}, torch::TensorOptions().device(cpu_device).dtype(torch::kFloat32));
  torch::Tensor cos_sin_qc_cache = torch::rand({chunk_len, rotary_dim}, torch::TensorOptions().device(cpu_device).dtype(torch::kFloat32));
  torch::Tensor cos_sin_qc_no_clamp_cache = torch::rand({chunk_len, rotary_dim}, torch::TensorOptions().device(cpu_device).dtype(torch::kFloat32));
  torch::Tensor cos_sin_q_inter_cache = torch::rand({chunk_len, rotary_dim}, torch::TensorOptions().device(cpu_device).dtype(torch::kFloat32));
  torch::Tensor out = torch::rand({total_length, head_size * 5}, torch::TensorOptions().device(cpu_device).dtype(torch::kFloat32));

  float* query_ptr = query.data_ptr<float>();
  float* key_ptr = key.data_ptr<float>();
  float* cos_sin_q_cache_ptr = cos_sin_q_cache.data_ptr<float>();
  float* cos_sin_qc_cache_ptr = cos_sin_qc_cache.data_ptr<float>();
  float* cos_sin_qc_no_clamp_cache_ptr = cos_sin_qc_no_clamp_cache.data_ptr<float>();
  float* cos_sin_q_inter_cache_ptr = cos_sin_q_inter_cache.data_ptr<float>();
  float* out_ptr = out.data_ptr<float>();

  std::cout << out_ptr[0] << std::endl;

  std::cout << cos_sin_q_cache_ptr[0] << ", " << cos_sin_q_cache_ptr[rotary_dim / 2] << ", "
            << query_ptr[0] << ", " << query_ptr[head_size / 2] << ", "
            << query_ptr[0] * cos_sin_q_cache_ptr[0] - query_ptr[head_size / 2] * cos_sin_q_cache_ptr[rotary_dim / 2] << ", "
            << std::endl;
  
  std::cout << cos_sin_qc_cache_ptr[0] << ", " << cos_sin_qc_cache_ptr[rotary_dim / 2] << ", "
            << query_ptr[0] << ", " << query_ptr[head_size / 2] << ", "
            << query_ptr[0] * cos_sin_qc_cache_ptr[0] - query_ptr[head_size / 2] * cos_sin_qc_cache_ptr[rotary_dim / 2] << ", "
            << std::endl;

  std::cout << cos_sin_qc_cache_ptr[(chunk_len - 1) * rotary_dim] << ", " << cos_sin_qc_cache_ptr[(chunk_len - 1) * rotary_dim + rotary_dim / 2] << ", "
            << query_ptr[0] << ", " << query_ptr[head_size / 2] << ", "
            << query_ptr[0] * cos_sin_qc_cache_ptr[(chunk_len - 1) * rotary_dim] - query_ptr[head_size / 2] * cos_sin_qc_cache_ptr[(chunk_len - 1) * rotary_dim + rotary_dim / 2] << ", "
            << std::endl;
  
  std::cout << cos_sin_qc_no_clamp_cache_ptr[0] << ", " << cos_sin_qc_no_clamp_cache_ptr[rotary_dim / 2] << ", "
            << query_ptr[0] << ", " << query_ptr[head_size / 2] << ", "
            << query_ptr[0] * cos_sin_qc_no_clamp_cache_ptr[0] - query_ptr[head_size / 2] * cos_sin_qc_no_clamp_cache_ptr[rotary_dim / 2] << ", "
            << std::endl;
  
  std::cout << cos_sin_q_inter_cache_ptr[0] << ", " << cos_sin_q_inter_cache_ptr[rotary_dim / 2] << ", "
            << query_ptr[0] << ", " << query_ptr[head_size / 2] << ", "
            << query_ptr[0] * cos_sin_q_inter_cache_ptr[0] - query_ptr[head_size / 2] * cos_sin_q_inter_cache_ptr[rotary_dim / 2] << ", "
            << std::endl;
  // std::cout << query << std::endl;
  // std::cout << key << std::endl;
  // std::cout << cos_sin_q_cache[0] << std::endl;
  // std::cout << cos_sin_qc_cache[0] << std::endl;
  // std::cout << cos_sin_qc_cache[chunk_len - 1] << std::endl;
  // std::cout << cos_sin_qc_no_clamp_cache[0] << std::endl;
  // std::cout << cos_sin_q_inter_cache[0] << std::endl;

  query = query.to(device);
  key = key.to(device);
  cos_sin_q_cache = cos_sin_q_cache.to(device);
  cos_sin_qc_cache = cos_sin_qc_cache.to(device);
  cos_sin_qc_no_clamp_cache = cos_sin_qc_no_clamp_cache.to(device);
  cos_sin_q_inter_cache = cos_sin_q_inter_cache.to(device);
  out = out.to(device);

  cudaDeviceSynchronize();
  dca_rotary_embedding(
    positions,
    query,
    key,
    head_size,
    cos_sin_q_cache,
    cos_sin_qc_cache,
    cos_sin_qc_no_clamp_cache,
    cos_sin_q_inter_cache,
    out,
    chunk_len,
    is_neox
  );
  cudaDeviceSynchronize();
  std::cout << out.size(0) << ", " << out.size(1) << std::endl;

  // std::cout << query.cpu() << std::endl;
  // std::cout << key.cpu() << std::endl;
  // std::cout << cos_sin_q_cache.cpu()[0] << std::endl;
  // std::cout << cos_sin_qc_cache.cpu()[0] << std::endl;
  // std::cout << cos_sin_qc_cache.cpu()[chunk_len - 1] << std::endl;
  // std::cout << cos_sin_qc_no_clamp_cache.cpu()[0] << std::endl;
  // std::cout << cos_sin_q_inter_cache.cpu()[0] << std::endl;
  // std::cout << out.cpu() << std::endl;
  torch::Tensor out_cpu = out.cpu();
  // query.print();
  // key.print();
  // cos_sin_q_cache.print();
  // cos_sin_qc_cache.print();
  // cos_sin_qc_no_clamp_cache.print();
  // cos_sin_q_inter_cache.print();
  // out.print();
  // out_ptr = out.to(cpu_device).data_ptr<float>();
  // std::cout << out_ptr[0] << ", " << out_ptr[head_size] << ", " << out_ptr[head_size * 2] << ", " << out_ptr[head_size * 3] << ", " << out_ptr[head_size * 4] << std::endl;

  out_ptr = out_cpu.data_ptr<float>();
  std::cout << out_ptr[0] << ", " << out_ptr[head_size] << ", " << out_ptr[head_size * 2] << ", " << out_ptr[head_size * 3] << ", " << out_ptr[head_size * 4] << std::endl;
  return 0;
}

// int main() {
//   torch::Device device(torch::kCPU);
//   torch::Tensor a = torch::tensor({1,2,3,4,5,6}, torch::TensorOptions().device(device));
//   // std::cout << a.cpu() << std::endl;

//   std::vector<int64_t> split_sizes = {3, 2, 1};
//   auto split_tensors = torch::split(a, split_sizes, 0);
//   for (size_t i = 0; i < split_tensors.size(); ++i) {
//     std::cout << split_tensors[i].storage_offset() << std::endl;
//     int64_t* ptr = split_tensors[i].data_ptr<int64_t>();
//     std::cout << "first val: " << *ptr << std::endl;
//   }
//   // torch::Tensor tensor = torch::randn({6, 3});
//   // std::cout << "原始张量形状: " << tensor.sizes() << std::endl;

//   // // 指定分割大小为 [2, 3, 1]，总和为 6
//   // std::vector<int64_t> split_sizes = {2, 3, 1};
//   // auto split_tensors = torch::split(tensor, split_sizes, 0);

//   // for (size_t i = 0; i < split_tensors.size(); ++i) {
//   //     std::cout << "分割后的张量 " << i << " 形状: " << split_tensors[i].sizes() << std::endl;
//   // }

//   return 0;
// }
