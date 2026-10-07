CC = gcc
CFLAGS ?= -O3 -march=native -ffast-math -std=c11 -static-libgcc
WIN_LIBS = -lm -ld3d12 -ld3dcompiler -ldxgi

.PHONY: all clean

all: atlas_1_0_chat.exe atlas_compact_chat.exe atlas_chain_monitor.exe atlas_1_0_infer.exe atlas_teacher_infer.exe atlas_compact_infer.exe atlas_dynamic_infer.exe specialist_loader.exe atlas_model_inspect.exe atlas_bridge_test.exe atlas_orchestrator_test.exe

atlas_1_0_chat.exe: src/chat_gui.c src/atlas_orchestrator.c src/atlas_orchestrator.h src/atlas_model_bridge.c src/atlas_model_registry.c src/atlas_conversation.c
	$(CC) $(CFLAGS) src/chat_gui.c src/atlas_orchestrator.c src/atlas_model_bridge.c src/atlas_model_registry.c src/atlas_conversation.c -o $@ -mwindows -luser32 -lgdi32 -lkernel32 -lshell32

atlas_compact_chat.exe: src/chat_gui.c
	$(CC) $(CFLAGS) -DATLAS_COMPACT $< -o $@ -mwindows -luser32 -lgdi32 -lkernel32 -lshell32

atlas_chain_monitor.exe: src/atlas_chain_monitor.c
	$(CC) $(CFLAGS) src/atlas_chain_monitor.c -o $@ -mwindows -luser32 -lgdi32 -lkernel32 -lshell32

atlas_1_0_infer.exe: src/brain_runtime.c src/gpu_matvec.c src/gpu_matvec.h
	$(CC) $(CFLAGS) -DATLAS_COMPACT -municode src/brain_runtime.c src/gpu_matvec.c -o $@ $(WIN_LIBS)

atlas_teacher_infer.exe: src/brain_runtime.c src/gpu_matvec.c src/gpu_matvec.h
	$(CC) $(CFLAGS) -municode src/brain_runtime.c src/gpu_matvec.c -o $@ $(WIN_LIBS)

atlas_compact_infer.exe: src/brain_runtime.c src/gpu_matvec.c src/gpu_matvec.h
	$(CC) $(CFLAGS) -DATLAS_COMPACT -municode src/brain_runtime.c src/gpu_matvec.c -o $@ $(WIN_LIBS)

atlas_dynamic_infer.exe: src/brain_runtime.c src/gpu_matvec.c src/gpu_matvec.h
	$(CC) $(CFLAGS) -municode src/brain_runtime.c src/gpu_matvec.c -o $@ $(WIN_LIBS)

atlas_tiny_135m_infer.exe: src/brain_runtime.c src/gpu_matvec.c src/gpu_matvec.h
	$(CC) $(CFLAGS) -DATLAS_TINY_135M -municode src/brain_runtime.c src/gpu_matvec.c -o $@ $(WIN_LIBS)

atlas_tiny_360m_infer.exe: src/brain_runtime.c src/gpu_matvec.c src/gpu_matvec.h
	$(CC) $(CFLAGS) -DATLAS_TINY_360M -municode src/brain_runtime.c src/gpu_matvec.c -o $@ $(WIN_LIBS)

specialist_loader.exe: src/specialist_loader.c src/atlas_model_config.c src/atlas_model_config.h
	$(CC) -O2 -std=c11 -static-libgcc -municode src/specialist_loader.c src/atlas_model_config.c -o $@

atlas_model_inspect.exe: src/atlas_model_config.c src/atlas_model_config.h
	$(CC) $(CFLAGS) src/atlas_model_config.c -o $@ -lm

atlas_bridge_test.exe: src/atlas_bridge_test.c src/atlas_model_bridge.c src/atlas_model_bridge.h src/atlas_conversation.c src/atlas_conversation.h
	$(CC) $(CFLAGS) src/atlas_bridge_test.c src/atlas_model_bridge.c src/atlas_conversation.c -o $@ -lm

atlas_orchestrator_test.exe: src/atlas_orchestrator_test.c src/atlas_orchestrator.c src/atlas_model_bridge.c src/atlas_conversation.c
	$(CC) $(CFLAGS) src/atlas_orchestrator_test.c src/atlas_orchestrator.c src/atlas_model_bridge.c src/atlas_conversation.c -o $@ -lm

clean:
	rm -f atlas_1_0_chat.exe atlas_compact_chat.exe atlas_chain_monitor.exe atlas_1_0_infer.exe atlas_teacher_infer.exe atlas_compact_infer.exe atlas_dynamic_infer.exe atlas_tiny_135m_infer.exe atlas_tiny_360m_infer.exe atlas_model_inspect.exe specialist_loader.exe atlas_bridge_test.exe atlas_orchestrator_test.exe
