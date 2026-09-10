#include "internal.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cmath>
#include <limits>
namespace apxinf::gemm {
struct Events{cudaEvent_t start=nullptr,stop=nullptr;Events(){check_cuda(cudaEventCreate(&start));check_cuda(cudaEventCreate(&stop));}~Events(){if(start)cudaEventDestroy(start);if(stop)cudaEventDestroy(stop);}};
struct Allocation{void* p=nullptr;explicit Allocation(size_t bytes){check_cuda(cudaMalloc(&p,bytes));}~Allocation(){if(p)cudaFree(p);}};
static std::vector<float> read(void* ptr,size_t count,int dtype,cudaStream_t stream){
 std::vector<unsigned char> data(count*dtype_bytes(dtype));check_cuda(cudaMemcpyAsync(data.data(),ptr,data.size(),cudaMemcpyDeviceToHost,stream));check_cuda(cudaStreamSynchronize(stream));std::vector<float> result(count);
 for(size_t i=0;i<count;i++){
  if(dtype==0)memcpy(&result[i],data.data()+4*i,4);
  else if(dtype==1){half v;memcpy(&v,data.data()+2*i,2);result[i]=__half2float(v);}
  else if(dtype==2){__nv_bfloat16 v;memcpy(&v,data.data()+2*i,2);result[i]=__bfloat162float(v);}
  else{__nv_fp8_e4m3 v;memcpy(&v,data.data()+i,1);result[i]=float(v);}
 }return result;
}
static void poison(const apxinf_gemm_bindings_t& b,size_t bytes){auto stream=(cudaStream_t)b.stream;check_cuda(cudaMemsetAsync(b.output,0xff,bytes,stream));check_cuda(cudaStreamSynchronize(stream));}
static bool valid_output(const std::vector<float>& expected,const std::vector<float>& actual,int dtype){double error=0,norm=0;for(size_t i=0;i<expected.size();i++){if(!std::isfinite(expected[i])||!std::isfinite(actual[i]))return false;double d=double(actual[i])-expected[i];error+=d*d;norm+=double(expected[i])*expected[i];}double tolerance=dtype==3?0.035:0.012;return std::sqrt(error/std::max(norm,1e-20))<=tolerance;}
static void verify_graph(State& candidate,const apxinf_gemm_bindings_t& b,const std::vector<float>& expected,size_t bytes,int dtype){
 auto stream=(cudaStream_t)b.stream;cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;
 check_cuda(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
 try{check_cuda(candidate.implementation->launch(candidate,b));}catch(...){cudaStreamEndCapture(stream,&graph);if(graph)cudaGraphDestroy(graph);throw;}
 check_cuda(cudaStreamEndCapture(stream,&graph));
 auto status=cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0);
 if(status!=cudaSuccess){cudaGraphDestroy(graph);check_cuda(status);}
 poison(b,bytes);status=cudaGraphLaunch(exec,stream);if(status==cudaSuccess)status=cudaStreamSynchronize(stream);
 if(status==cudaSuccess&&!valid_output(expected,read(b.output,expected.size(),dtype,stream),dtype)){status=cudaErrorLaunchFailure;}
 cudaGraphExecDestroy(exec);cudaGraphDestroy(graph);check_cuda(status);
}
std::shared_ptr<State> tune(const Spec& spec,const apxinf_gemm_policy_t& policy,const apxinf_gemm_bindings_t& bindings,int device,std::string& report){
 size_t count=spec.m*(spec.semantic==APXINF_GEMM_SEMANTIC_GEMM_GEGLU?spec.n/2:spec.n);
 Allocation out(count*dtype_bytes(spec.output_dtype));auto b=bindings;b.output=out.p;
 size_t output_bytes=count*dtype_bytes(spec.output_dtype);
 auto reference_policy=policy;reference_policy.workspace_limit=UINT64_MAX;
 const auto& implementations=registry(spec.semantic);
 auto reference=prepare(implementations.front(),0,spec,reference_policy,device);
 check_cuda(reference->implementation->launch(*reference,b));auto expected=read(out.p,count,spec.output_dtype,(cudaStream_t)b.stream);
 std::shared_ptr<State> winner;float best=std::numeric_limits<float>::infinity();Events events;int checked=0,rejected=0;std::vector<std::string> diagnostics;
 for(const auto& impl:implementations){
  if(!impl.supports(spec)){diagnostics.push_back(std::string(impl.name)+"=skip(contract)");continue;}
  if(!supports_alignment(impl,spec)){diagnostics.push_back(std::string(impl.name)+"=skip(alignment)");continue;}
  if(policy.graph_safe&&!impl.graph_safe){diagnostics.push_back(std::string(impl.name)+"=skip(graph-safe)");continue;}
  if(policy.deterministic&&!impl.deterministic){diagnostics.push_back(std::string(impl.name)+"=skip(determinism)");continue;}
  std::vector<int> configs;impl.enumerate_configs(spec,configs);
  for(int config:configs){
   try{
    auto candidate=prepare(impl,config,spec,policy,device);
    poison(b,output_bytes);check_cuda(impl.launch(*candidate,b));auto actual=read(out.p,count,spec.output_dtype,(cudaStream_t)b.stream);
    bool valid=valid_output(expected,actual,spec.output_dtype);
    ++checked;if(!valid){++rejected;diagnostics.push_back(std::string(impl.name)+"#"+std::to_string(config)+"=reject(numeric)");continue;}
    if(policy.graph_safe)verify_graph(*candidate,b,expected,output_bytes,spec.output_dtype);
    for(int i=0;i<3;i++)check_cuda(impl.launch(*candidate,b));
    check_cuda(cudaEventRecord(events.start,(cudaStream_t)b.stream));
    for(int i=0;i<10;i++)check_cuda(impl.launch(*candidate,b));
    check_cuda(cudaEventRecord(events.stop,(cudaStream_t)b.stream));check_cuda(cudaEventSynchronize(events.stop));
    float ms;check_cuda(cudaEventElapsedTime(&ms,events.start,events.stop));ms/=10;
    diagnostics.push_back(std::string(impl.name)+"#"+std::to_string(config)+"=pass");
    if(ms<best){best=ms;winner=candidate;}
   }catch(const Failure& failure){++rejected;diagnostics.push_back(std::string(impl.name)+"#"+std::to_string(config)+"=reject("+failure.what()+")");cudaGetLastError();}
  }
 }
 if(!winner)throw Failure(APXINF_STATUS_UNSUPPORTED,"no validated GEMM candidate");
 report="tuned checked="+std::to_string(checked)+" rejected="+std::to_string(rejected)+" ms="+std::to_string(best)+" candidates=[";
 for(size_t i=0;i<diagnostics.size();++i){if(i!=0)report+=",";report+=diagnostics[i];}
 report+="]";
 return winner;
}
}
