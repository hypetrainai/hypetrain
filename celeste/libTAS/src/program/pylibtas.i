%module pylibtas
%begin %{
#define SWIG_PYTHON_STRICT_BYTE_CHAR
%}

%{
#include "../shared/SharedConfig.h"
#include "GameLoop.h"
%}

%include "std_string.i"

void launchGameThread(
    const std::string& gamepath,
    const std::string& libtaspath,
    const std::string& gameargs = "",
    int startframe = 0,
    const std::string& libdir = "",
    const std::string& rundir = "",
    int logging_status = SharedConfig::LOGGING_TO_CONSOLE,
    bool opengl_soft = true,
    const std::string& llvm_perf = "",
    bool attach_gdb = false);
