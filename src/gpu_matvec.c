#define WIN32_LEAN_AND_MEAN
#define COBJMACROS
#define INITGUID
#include <windows.h>
#include <d3d12.h>
#include <d3dcompiler.h>
#include <dxgi1_4.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gpu_matvec.h"

typedef struct {
    ID3D12Device *device;
    ID3D12CommandQueue *queue;
    ID3D12CommandAllocator *allocator;
    ID3D12GraphicsCommandList *list;
    ID3D12Fence *fence;
    ID3D12RootSignature *root;
    ID3D12PipelineState *pipeline;
    ID3D12DescriptorHeap *heap;
    ID3D12Resource *model, *model_upload, *input, *output, *readback;
    uint8_t *model_map;
    float *input_map, *readback_map;
    UINT descriptor_step;
    HANDLE event;
    UINT64 fence_value;
    size_t model_size;
    uint64_t max_output;
    int ready;
} Gpu;

static Gpu gpu;

static const char shader_source[] =
"ByteAddressBuffer modelData : register(t0);\n"
"StructuredBuffer<float> inputData : register(t1);\n"
"RWStructuredBuffer<float> outputData : register(u0);\n"
"cbuffer Params : register(b0) { uint dataOffset; uint rows; uint cols; uint quantType; uint inputOffset; uint outputOffset; };\n"
"uint byteAt(uint a) { uint v=modelData.Load(a & ~3u); return (v >> ((a & 3u)*8)) & 255u; }\n"
"float halfAt(uint a) { return f16tof32(byteAt(a)|(byteAt(a+1u)<<8)); }\n"
"void scaleMin(uint b, uint j, out uint sc, out uint mn) {\n"
" if (j<4) { sc=byteAt(b+4+j)&63u; mn=byteAt(b+8+j)&63u; }\n"
" else { sc=(byteAt(b+4+j+4)&15u)|((byteAt(b+4+j-4)>>6)<<4); mn=(byteAt(b+4+j+4)>>4)|((byteAt(b+4+j)>>6)<<4); } }\n"
"float weightAt(uint block, uint i) {\n"
" if (quantType==12u) { uint seg=i/64u, sub=i%64u, si=seg*2u+(sub>=32u?1u:0u), sc,mn; scaleMin(block,si,sc,mn); uint qbyte=byteAt(block+16u+seg*32u+(sub%32u)); uint q=(sub<32u)?(qbyte&15u):(qbyte>>4); return halfAt(block)*(float)sc*(float)q-halfAt(block+2u)*(float)mn; }\n"
" if (quantType==13u) { uint seg=i/64u, sub=i%64u, si=seg*2u+(sub>=32u?1u:0u), sc,mn; scaleMin(block,si,sc,mn); uint lane=sub%32u; uint qbyte=byteAt(block+48u+seg*32u+lane); uint q=(sub<32u)?(qbyte&15u):(qbyte>>4); uint high=byteAt(block+16u+seg*8u+lane); uint mask=1u<<(2u*seg+(sub>=32u?1u:0u)); q|=((high&mask)!=0u)?16u:0u; return halfAt(block)*(float)sc*(float)q-halfAt(block+2u)*(float)mn; }\n"
" if (quantType==14u) { uint halfIndex=i/128u, p=i%128u, chunk=p/32u, lane=p%32u; uint laneByte=byteAt(block+halfIndex*64u+(chunk&1u)*32u+lane); uint lo=(chunk<2u)?(laneByte&15u):(laneByte>>4); uint high=byteAt(block+128u+halfIndex*32u+lane); uint hi=(high>>(chunk*2u))&3u; uint scaleIndex=halfIndex*8u+chunk*2u+(lane>=16u?1u:0u); int sc=(int)byteAt(block+192u+scaleIndex); if(sc>=128) sc-=256; int q=(int)(lo|(hi<<4))-32; return halfAt(block+208u)*(float)sc*(float)q; }\n"
" if (quantType==6u) { uint qbyte=byteAt(block+6u+(i&15u)); uint q=(i<16u)?(qbyte&15u):(qbyte>>4); uint high=byteAt(block+2u)|(byteAt(block+3u)<<8u)|(byteAt(block+4u)<<16u)|(byteAt(block+5u)<<24u); q|=((high>>i)&1u)<<4; return halfAt(block)*(float)((int)q-16); }\n"
" if (quantType==7u) { uint qbyte=byteAt(block+8u+(i&15u)); uint q=(i<16u)?(qbyte&15u):(qbyte>>4); uint high=byteAt(block+4u)|(byteAt(block+5u)<<8u)|(byteAt(block+6u)<<16u)|(byteAt(block+7u)<<24u); q|=((high>>i)&1u)<<4; return halfAt(block)*(float)q+halfAt(block+2u); }\n"
" if (quantType==8u) { int q=(int)byteAt(block+2u+i); if(q>=128) q-=256; return halfAt(block)*(float)q; }\n"
" return 0.0; }\n"
"groupshared float partial[256];\n"
"[numthreads(256,1,1)] void main(uint3 gid : SV_GroupID, uint3 laneId : SV_GroupThreadID) { uint group=gid.x+gid.y*65535u; uint slot=laneId.x/64u; uint lane=laneId.x%64u; uint r=group*4u+slot; uint be=(quantType==12u||quantType==13u||quantType==14u)?256u:32u; uint bs=quantType==6u?22u:quantType==7u?24u:quantType==8u?34u:quantType==12u?144u:quantType==13u?176u:210u; float sum=0.0; if(r<rows) for(uint i=lane;i<cols;i+=64u) { uint b=i/be, j=i%be; uint base=dataOffset+(r*(cols/be)+b)*bs; sum=mad(weightAt(base,j),inputData[inputOffset+i],sum); } uint at=slot*64u+lane; partial[at]=sum; GroupMemoryBarrierWithGroupSync(); for(uint stride=32u;stride>0u;stride>>=1u) { if(lane<stride) partial[at]+=partial[at+stride]; GroupMemoryBarrierWithGroupSync(); } if(lane==0u && r<rows) outputData[outputOffset+r]=partial[at]; }\n";

static int create_buffer(D3D12_HEAP_TYPE heap_type, UINT64 size,
                         D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state,
                         ID3D12Resource **out)
{
    D3D12_HEAP_PROPERTIES hp = {0};
    hp.Type = heap_type; hp.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    hp.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN; hp.CreationNodeMask = 1; hp.VisibleNodeMask = 1;
    D3D12_RESOURCE_DESC desc = {0};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; desc.Width = (size + 255u) & ~255ull;
    desc.Height = 1; desc.DepthOrArraySize = 1; desc.MipLevels = 1;
    desc.SampleDesc.Count = 1; desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; desc.Flags = flags;
    return SUCCEEDED(ID3D12Device_CreateCommittedResource(gpu.device, &hp,
        D3D12_HEAP_FLAG_NONE, &desc, state, NULL, &IID_ID3D12Resource, (void **)out));
}

void brain_gpu_shutdown(void)
{
    if (gpu.event) CloseHandle(gpu.event);
    if (gpu.readback) ID3D12Resource_Unmap(gpu.readback, 0, NULL);
    if (gpu.input) ID3D12Resource_Unmap(gpu.input, 0, NULL);
    if (gpu.model_upload) ID3D12Resource_Unmap(gpu.model_upload, 0, NULL);
    if (gpu.readback) ID3D12Resource_Release(gpu.readback);
    if (gpu.output) ID3D12Resource_Release(gpu.output);
    if (gpu.input) ID3D12Resource_Release(gpu.input);
    if (gpu.model_upload) ID3D12Resource_Release(gpu.model_upload);
    if (gpu.model) ID3D12Resource_Release(gpu.model);
    if (gpu.heap) ID3D12DescriptorHeap_Release(gpu.heap);
    if (gpu.pipeline) ID3D12PipelineState_Release(gpu.pipeline);
    if (gpu.root) ID3D12RootSignature_Release(gpu.root);
    if (gpu.fence) ID3D12Fence_Release(gpu.fence);
    if (gpu.list) ID3D12GraphicsCommandList_Release(gpu.list);
    if (gpu.allocator) ID3D12CommandAllocator_Release(gpu.allocator);
    if (gpu.queue) ID3D12CommandQueue_Release(gpu.queue);
    if (gpu.device) ID3D12Device_Release(gpu.device);
    memset(&gpu, 0, sizeof(gpu));
}

int brain_gpu_init(const uint8_t *model, size_t model_size)
{
    const char *failed_stage = "D3D12 device creation";
    brain_gpu_shutdown();
    if (!model || model_size < 4 || (model_size & 3u)) return 0;
    if (FAILED(D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_11_0,
                                 &IID_ID3D12Device, (void **)&gpu.device))) goto fail;
    failed_stage = "command queue and synchronization";
    D3D12_COMMAND_QUEUE_DESC qd = {0}; qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    if (FAILED(ID3D12Device_CreateCommandQueue(gpu.device, &qd, &IID_ID3D12CommandQueue, (void **)&gpu.queue))) goto fail;
    if (FAILED(ID3D12Device_CreateCommandAllocator(gpu.device, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                  &IID_ID3D12CommandAllocator, (void **)&gpu.allocator))) goto fail;
    if (FAILED(ID3D12Device_CreateCommandList(gpu.device, 0, D3D12_COMMAND_LIST_TYPE_DIRECT,
        gpu.allocator, NULL, &IID_ID3D12GraphicsCommandList, (void **)&gpu.list))) goto fail;
    ID3D12GraphicsCommandList_Close(gpu.list);
    if (FAILED(ID3D12Device_CreateFence(gpu.device, 0, D3D12_FENCE_FLAG_NONE,
                                        &IID_ID3D12Fence, (void **)&gpu.fence))) goto fail;
    gpu.event = CreateEventW(NULL, FALSE, FALSE, NULL); if (!gpu.event) goto fail;

    failed_stage = "root signature and shader compilation";
    D3D12_DESCRIPTOR_RANGE ranges[2] = {0};
    ranges[0].RangeType=D3D12_DESCRIPTOR_RANGE_TYPE_SRV; ranges[0].NumDescriptors=2;
    ranges[0].BaseShaderRegister=0; ranges[0].OffsetInDescriptorsFromTableStart=D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
    ranges[1].RangeType=D3D12_DESCRIPTOR_RANGE_TYPE_UAV; ranges[1].NumDescriptors=1;
    ranges[1].BaseShaderRegister=0; ranges[1].OffsetInDescriptorsFromTableStart=D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
    D3D12_ROOT_PARAMETER params[3] = {0};
    params[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[0].DescriptorTable.NumDescriptorRanges=1; params[0].DescriptorTable.pDescriptorRanges=&ranges[0];
    params[0].ShaderVisibility=D3D12_SHADER_VISIBILITY_ALL;
    params[1].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable.NumDescriptorRanges=1; params[1].DescriptorTable.pDescriptorRanges=&ranges[1];
    params[1].ShaderVisibility=D3D12_SHADER_VISIBILITY_ALL;
    params[2].ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[2].Constants.ShaderRegister=0; params[2].Constants.Num32BitValues=6;
    params[2].ShaderVisibility=D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC rsd={0}; rsd.NumParameters=3; rsd.pParameters=params;
    rsd.Flags=D3D12_ROOT_SIGNATURE_FLAG_NONE;
    ID3DBlob *blob=NULL,*errors=NULL;
    if (FAILED(D3D12SerializeRootSignature(&rsd,D3D_ROOT_SIGNATURE_VERSION_1,&blob,&errors))) {
        if (errors) { OutputDebugStringA((const char *)ID3D10Blob_GetBufferPointer(errors)); ID3D10Blob_Release(errors); }
        goto fail;
    }
    HRESULT hr=ID3D12Device_CreateRootSignature(gpu.device,0,ID3D10Blob_GetBufferPointer(blob),
        ID3D10Blob_GetBufferSize(blob),&IID_ID3D12RootSignature,(void **)&gpu.root);
    ID3D10Blob_Release(blob); if (FAILED(hr)) goto fail;
    ID3DBlob *code=NULL;
    hr=D3DCompile(shader_source,sizeof(shader_source)-1,"atlas_1_0_matvec.hlsl",NULL,NULL,
                  "main","cs_5_0",D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&code,&errors);
    if (FAILED(hr)) {
        if (errors) { OutputDebugStringA((const char *)ID3D10Blob_GetBufferPointer(errors)); ID3D10Blob_Release(errors); }
        goto fail;
    }
    D3D12_COMPUTE_PIPELINE_STATE_DESC pd={0}; pd.pRootSignature=gpu.root;
    pd.CS.pShaderBytecode=ID3D10Blob_GetBufferPointer(code); pd.CS.BytecodeLength=ID3D10Blob_GetBufferSize(code);
    hr=ID3D12Device_CreateComputePipelineState(gpu.device,&pd,&IID_ID3D12PipelineState,(void **)&gpu.pipeline);
    ID3D10Blob_Release(code); if (FAILED(hr)) goto fail;
    failed_stage = "GPU buffer allocation and mapping";

    const UINT64 input_size=26880ull*sizeof(float), output_size=151936ull*sizeof(float);
    if (!create_buffer(D3D12_HEAP_TYPE_DEFAULT,model_size,D3D12_RESOURCE_FLAG_NONE,D3D12_RESOURCE_STATE_COPY_DEST,&gpu.model) ||
        !create_buffer(D3D12_HEAP_TYPE_UPLOAD,model_size,D3D12_RESOURCE_FLAG_NONE,D3D12_RESOURCE_STATE_GENERIC_READ,&gpu.model_upload) ||
        !create_buffer(D3D12_HEAP_TYPE_UPLOAD,input_size,D3D12_RESOURCE_FLAG_NONE,D3D12_RESOURCE_STATE_GENERIC_READ,&gpu.input) ||
        !create_buffer(D3D12_HEAP_TYPE_DEFAULT,output_size,D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,&gpu.output) ||
        !create_buffer(D3D12_HEAP_TYPE_READBACK,output_size,D3D12_RESOURCE_FLAG_NONE,D3D12_RESOURCE_STATE_COPY_DEST,&gpu.readback)) goto fail;
    D3D12_RANGE no_read={0,0}; void *ptr=NULL;
    if (FAILED(ID3D12Resource_Map(gpu.model_upload,0,&no_read,&ptr))) goto fail;
    gpu.model_map=ptr; memcpy(gpu.model_map,model,model_size); gpu.model_size=model_size;
    if (FAILED(ID3D12Resource_Map(gpu.input,0,&no_read,&ptr))) goto fail;
    gpu.input_map=ptr;
    D3D12_RANGE read_range={0,(SIZE_T)output_size};
    if (FAILED(ID3D12Resource_Map(gpu.readback,0,&read_range,&ptr))) goto fail;
    gpu.readback_map=ptr;

    if (FAILED(ID3D12CommandAllocator_Reset(gpu.allocator)) ||
        FAILED(ID3D12GraphicsCommandList_Reset(gpu.list,gpu.allocator,NULL))) goto fail;
    ID3D12GraphicsCommandList_CopyBufferRegion(gpu.list,gpu.model,0,gpu.model_upload,0,model_size);
    D3D12_RESOURCE_BARRIER model_barrier={0}; model_barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    model_barrier.Transition.pResource=gpu.model; model_barrier.Transition.Subresource=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    model_barrier.Transition.StateBefore=D3D12_RESOURCE_STATE_COPY_DEST;
    model_barrier.Transition.StateAfter=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    ID3D12GraphicsCommandList_ResourceBarrier(gpu.list,1,&model_barrier);
    if (FAILED(ID3D12GraphicsCommandList_Close(gpu.list))) goto fail;
    ID3D12CommandList *upload_lists[]={(ID3D12CommandList *)gpu.list};
    ID3D12CommandQueue_ExecuteCommandLists(gpu.queue,1,upload_lists);
    if (FAILED(ID3D12CommandQueue_Signal(gpu.queue,gpu.fence,++gpu.fence_value)) ||
        FAILED(ID3D12Fence_SetEventOnCompletion(gpu.fence,gpu.fence_value,gpu.event)) ||
        WaitForSingleObject(gpu.event,INFINITE)!=WAIT_OBJECT_0) goto fail;
    ID3D12Resource_Unmap(gpu.model_upload,0,NULL); gpu.model_map=NULL;
    ID3D12Resource_Release(gpu.model_upload); gpu.model_upload=NULL;

    D3D12_DESCRIPTOR_HEAP_DESC hd={0}; hd.Type=D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    hd.NumDescriptors=3; hd.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (FAILED(ID3D12Device_CreateDescriptorHeap(gpu.device,&hd,&IID_ID3D12DescriptorHeap,(void **)&gpu.heap))) goto fail;
    failed_stage = "descriptor and model upload";
    gpu.descriptor_step=ID3D12Device_GetDescriptorHandleIncrementSize(gpu.device,D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_CPU_DESCRIPTOR_HANDLE cpu;
    gpu.heap->lpVtbl->GetCPUDescriptorHandleForHeapStart(gpu.heap,&cpu);
    D3D12_SHADER_RESOURCE_VIEW_DESC srv={0}; srv.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.ViewDimension=D3D12_SRV_DIMENSION_BUFFER; srv.Format=DXGI_FORMAT_R32_TYPELESS;
    srv.Buffer.NumElements=(UINT)(model_size/4); srv.Buffer.Flags=D3D12_BUFFER_SRV_FLAG_RAW;
    ID3D12Device_CreateShaderResourceView(gpu.device,gpu.model,&srv,cpu);
    cpu.ptr+=gpu.descriptor_step; srv.Format=DXGI_FORMAT_UNKNOWN; srv.Buffer.NumElements=26880;
    srv.Buffer.Flags=D3D12_BUFFER_SRV_FLAG_NONE; srv.Buffer.StructureByteStride=sizeof(float);
    ID3D12Device_CreateShaderResourceView(gpu.device,gpu.input,&srv,cpu);
    cpu.ptr+=gpu.descriptor_step;
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav={0}; uav.ViewDimension=D3D12_UAV_DIMENSION_BUFFER;
    uav.Format=DXGI_FORMAT_UNKNOWN; uav.Buffer.NumElements=151936; uav.Buffer.StructureByteStride=sizeof(float);
    ID3D12Device_CreateUnorderedAccessView(gpu.device,gpu.output,NULL,&uav,cpu);
    gpu.max_output=151936; gpu.ready=1;
    return 1;
fail:
    fprintf(stderr, "Direct3D 12 initialization failed during %s.\n", failed_stage);
    brain_gpu_shutdown(); return 0;
}

int brain_gpu_matvec_batch(const BrainGpuMatvec *ops,size_t count)
{
    static LONG reported = 0;
#define GPU_FAILURE(stage, hr) do { \
    if (InterlockedIncrement(&reported) <= 4) { \
        fprintf(stderr,"GPU matvec batch %s failed (0x%08lx), operations=%llu\n",stage,(unsigned long)(hr),(unsigned long long)count); \
    } \
    return 0; \
} while (0)
    if (!gpu.ready || !ops || !count || count>3) GPU_FAILURE("arguments", E_INVALIDARG);
    UINT input_offsets[3], output_offsets[3], input_total=0, output_total=0;
    for (size_t i=0;i<count;++i) {
        const BrainGpuMatvec *op=&ops[i];
        if (!op->input || !op->output || !op->rows || !op->cols ||
            op->rows>gpu.max_output || op->cols>8960 || op->offset>UINT32_MAX ||
            op->rows>UINT32_MAX || op->cols>UINT32_MAX ||
            (op->type!=6 && op->type!=7 && op->type!=8 && op->type!=12 && op->type!=13 && op->type!=14) ||
            input_total+op->cols>26880 || output_total+op->rows>gpu.max_output)
            GPU_FAILURE("arguments", E_INVALIDARG);
        input_offsets[i]=input_total; output_offsets[i]=output_total;
        memcpy(gpu.input_map+input_total,op->input,(size_t)op->cols*sizeof(float));
        input_total+=(UINT)op->cols; output_total+=(UINT)op->rows;
    }
    HRESULT hr=ID3D12CommandAllocator_Reset(gpu.allocator);
    if (FAILED(hr)) GPU_FAILURE("allocator reset",hr);
    hr=ID3D12GraphicsCommandList_Reset(gpu.list,gpu.allocator,gpu.pipeline);
    if (FAILED(hr)) GPU_FAILURE("list reset",hr);
    ID3D12DescriptorHeap *heaps[]={gpu.heap};
    ID3D12GraphicsCommandList_SetDescriptorHeaps(gpu.list,1,heaps);
    D3D12_GPU_DESCRIPTOR_HANDLE srv;
    gpu.heap->lpVtbl->GetGPUDescriptorHandleForHeapStart(gpu.heap,&srv);
    D3D12_GPU_DESCRIPTOR_HANDLE uav=srv; uav.ptr+=2ull*gpu.descriptor_step;
    ID3D12GraphicsCommandList_SetComputeRootSignature(gpu.list,gpu.root);
    ID3D12GraphicsCommandList_SetComputeRootDescriptorTable(gpu.list,0,srv);
    ID3D12GraphicsCommandList_SetComputeRootDescriptorTable(gpu.list,1,uav);
    for (size_t i=0;i<count;++i) {
        const BrainGpuMatvec *op=&ops[i];
        UINT constants[6]={(UINT)op->offset,(UINT)op->rows,(UINT)op->cols,
                           op->type,input_offsets[i],output_offsets[i]};
        ID3D12GraphicsCommandList_SetComputeRoot32BitConstants(gpu.list,2,6,constants,0);
        UINT row_groups=(UINT)((op->rows+3u)/4u);
        UINT groups_x=(UINT)(row_groups<65535u?row_groups:65535u);
        UINT groups_y=(UINT)((row_groups+groups_x-1u)/groups_x);
        ID3D12GraphicsCommandList_Dispatch(gpu.list,groups_x,groups_y,1);
    }
    D3D12_RESOURCE_BARRIER barrier={0}; barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.UAV.pResource=gpu.output; ID3D12GraphicsCommandList_ResourceBarrier(gpu.list,1,&barrier);
    barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource=gpu.output;
    barrier.Transition.Subresource=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore=D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    barrier.Transition.StateAfter=D3D12_RESOURCE_STATE_COPY_SOURCE;
    ID3D12GraphicsCommandList_ResourceBarrier(gpu.list,1,&barrier);
    for (size_t i=0;i<count;++i)
        ID3D12GraphicsCommandList_CopyBufferRegion(gpu.list,gpu.readback,
            (UINT64)output_offsets[i]*sizeof(float),gpu.output,
            (UINT64)output_offsets[i]*sizeof(float),(UINT64)ops[i].rows*sizeof(float));
    barrier.Transition.StateBefore=D3D12_RESOURCE_STATE_COPY_SOURCE;
    barrier.Transition.StateAfter=D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    ID3D12GraphicsCommandList_ResourceBarrier(gpu.list,1,&barrier);
    hr=ID3D12GraphicsCommandList_Close(gpu.list);
    if (FAILED(hr)) GPU_FAILURE("list close",hr);
    ID3D12CommandList *lists[]={(ID3D12CommandList *)gpu.list};
    ID3D12CommandQueue_ExecuteCommandLists(gpu.queue,1,lists);
    UINT64 value=++gpu.fence_value;
    hr=ID3D12CommandQueue_Signal(gpu.queue,gpu.fence,value);
    if (FAILED(hr)) GPU_FAILURE("queue signal",hr);
    if (ID3D12Fence_GetCompletedValue(gpu.fence)<value) {
        hr=ID3D12Fence_SetEventOnCompletion(gpu.fence,value,gpu.event);
        if (FAILED(hr)) GPU_FAILURE("fence event",hr);
        if (WaitForSingleObject(gpu.event,INFINITE)!=WAIT_OBJECT_0) GPU_FAILURE("fence wait",GetLastError());
    }
    for (size_t i=0;i<count;++i)
        memcpy(ops[i].output,gpu.readback_map+output_offsets[i],
               (size_t)ops[i].rows*sizeof(float));
    return 1;
#undef GPU_FAILURE
}

int brain_gpu_matvec(uint64_t offset,uint32_t type,uint64_t rows,uint64_t cols,
                     const float *input,float *output)
{
    BrainGpuMatvec op={offset,rows,cols,type,input,output};
    return brain_gpu_matvec_batch(&op,1);
}
