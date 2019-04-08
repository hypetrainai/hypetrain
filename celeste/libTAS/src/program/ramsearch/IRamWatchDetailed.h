/*
    Copyright 2015-2018 Clément Gallet <clement.gallet@ens-lyon.org>

    This file is part of libTAS.

    libTAS is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    libTAS is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with libTAS.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef LIBTAS_IRAMWATCHDETAILED_H_INCLUDED
#define LIBTAS_IRAMWATCHDETAILED_H_INCLUDED

// #include <cstdint>
// #include <sys/types.h>
#include <string>
#include <vector>

class IRamWatchDetailed {
public:
    IRamWatchDetailed(uintptr_t addr) : address(addr) {};
    virtual ~IRamWatchDetailed() = default;

    /* Return the current value of the ram watch as a string */
    virtual std::string value_str() = 0;

    /* Poke a value (given as a string) into the ram watch address. Return
     * the result of process_vm_writev call
     */
    virtual int poke_value(std::string) = 0;

    /* Returns the index of the stored type */
    virtual int type() = 0;

    uintptr_t address;
    std::string label;
    bool hex;

    bool isPointer;
    std::vector<int> pointer_offsets;
    uintptr_t base_address;

    static pid_t game_pid;
    static bool isValid;

};

#endif
