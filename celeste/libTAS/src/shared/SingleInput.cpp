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

#include "SingleInput.h"
#include <linux/input.h>
#include <SDL2/SDL.h>
#include "../external/SDL1.h"
// #include <X11/keysym.h>
#include <xcb/xcb.h>

bool SingleInput::isAnalog() const
{
    if ((type == IT_POINTER_X) || (type == IT_POINTER_Y))
        return true;
    if (inputTypeIsController()) {
        return inputTypeToAxisFlag();
    }
    return false;
}

int SingleInput::inputTypeIsController() const
{
    return (type >= IT_CONTROLLER1_BUTTON_A) && (type <= IT_CONTROLLER4_AXIS_TRIGGERRIGHT);
}

int SingleInput::inputTypeToControllerNumber() const
{
    return (type >> IT_CONTROLLER_ID_SHIFT) - 1;
}

bool SingleInput::inputTypeToAxisFlag() const
{
    return type & IT_CONTROLLER_AXIS_MASK;
}

int SingleInput::inputTypeToInputNumber() const
{
    return type & IT_CONTROLLER_TYPE_MASK;
}

unsigned int SingleInput::toXlibPointerButton(int button)
{
    switch (button) {
        case SingleInput::POINTER_B1:
            return XCB_BUTTON_INDEX_1;
        case SingleInput::POINTER_B2:
            return XCB_BUTTON_INDEX_2;
        case SingleInput::POINTER_B3:
            return XCB_BUTTON_INDEX_3;
        case SingleInput::POINTER_B4:
            return XCB_BUTTON_INDEX_4;
        case SingleInput::POINTER_B5:
            return XCB_BUTTON_INDEX_5;
        default:
            return 0;
    }
}

unsigned int SingleInput::toSDL1PointerButton(int button)
{
    switch (button) {
        case SingleInput::POINTER_B1:
            return SDL1::SDL1_BUTTON_LEFT;
        case SingleInput::POINTER_B2:
            return SDL1::SDL1_BUTTON_MIDDLE;
        case SingleInput::POINTER_B3:
            return SDL1::SDL1_BUTTON_RIGHT;
        case SingleInput::POINTER_B4:
            return SDL1::SDL1_BUTTON_X1;
        case SingleInput::POINTER_B5:
            return SDL1::SDL1_BUTTON_X2;
        default:
            return 0;
    }
}

unsigned int SingleInput::toSDL2PointerButton(int button)
{
    switch (button) {
        case SingleInput::POINTER_B1:
            return SDL_BUTTON_LEFT;
        case SingleInput::POINTER_B2:
            return SDL_BUTTON_MIDDLE;
        case SingleInput::POINTER_B3:
            return SDL_BUTTON_RIGHT;
        case SingleInput::POINTER_B4:
            return SDL_BUTTON_X1;
        case SingleInput::POINTER_B5:
            return SDL_BUTTON_X2;
        default:
            return 0;
    }
}


unsigned int SingleInput::toXlibPointerMask(int mask)
{
    unsigned int xlib_mask = 0;
    if (mask & (1 << SingleInput::POINTER_B1))
        xlib_mask |= XCB_BUTTON_MASK_1;
    if (mask & (1 << SingleInput::POINTER_B2))
        xlib_mask |= XCB_BUTTON_MASK_2;
    if (mask & (1 << SingleInput::POINTER_B3))
        xlib_mask |= XCB_BUTTON_MASK_3;
    if (mask & (1 << SingleInput::POINTER_B4))
        xlib_mask |= XCB_BUTTON_MASK_4;
    if (mask & (1 << SingleInput::POINTER_B5))
        xlib_mask |= XCB_BUTTON_MASK_5;

    return xlib_mask;
}

unsigned int SingleInput::toSDL1PointerMask(int mask)
{
    unsigned int sdl_mask = 0;
    if (mask & (1 << SingleInput::POINTER_B1))
        sdl_mask |= SDL1::SDL1_BUTTON_LMASK;
    if (mask & (1 << SingleInput::POINTER_B2))
        sdl_mask |= SDL1::SDL1_BUTTON_MMASK;
    if (mask & (1 << SingleInput::POINTER_B3))
        sdl_mask |= SDL1::SDL1_BUTTON_RMASK;
    if (mask & (1 << SingleInput::POINTER_B4))
        sdl_mask |= SDL1::SDL1_BUTTON_X1MASK;
    if (mask & (1 << SingleInput::POINTER_B5))
        sdl_mask |= SDL1::SDL1_BUTTON_X2MASK;

    return sdl_mask;
}

unsigned int SingleInput::toSDL2PointerMask(int mask)
{
    unsigned int sdl_mask = 0;
    if (mask & (1 << SingleInput::POINTER_B1))
        sdl_mask |= SDL_BUTTON_LMASK;
    if (mask & (1 << SingleInput::POINTER_B2))
        sdl_mask |= SDL_BUTTON_MMASK;
    if (mask & (1 << SingleInput::POINTER_B3))
        sdl_mask |= SDL_BUTTON_RMASK;
    if (mask & (1 << SingleInput::POINTER_B4))
        sdl_mask |= SDL_BUTTON_X1MASK;
    if (mask & (1 << SingleInput::POINTER_B5))
        sdl_mask |= SDL_BUTTON_X2MASK;

    return sdl_mask;
}


int SingleInput::toSDL2Axis(int axis)
{
    switch(axis) {
        case SingleInput::AXIS_LEFTX:
            return SDL_CONTROLLER_AXIS_LEFTX;
        case SingleInput::AXIS_LEFTY:
            return SDL_CONTROLLER_AXIS_LEFTY;
        case SingleInput::AXIS_RIGHTX:
            return SDL_CONTROLLER_AXIS_RIGHTX;
        case SingleInput::AXIS_RIGHTY:
            return SDL_CONTROLLER_AXIS_RIGHTY;
        case SingleInput::AXIS_TRIGGERLEFT:
            return SDL_CONTROLLER_AXIS_TRIGGERLEFT;
        case SingleInput::AXIS_TRIGGERRIGHT:
            return SDL_CONTROLLER_AXIS_TRIGGERRIGHT;
        default:
            return SDL_CONTROLLER_AXIS_INVALID;
    }
}

int SingleInput::toSDL2Button(int button)
{
    switch(button) {
        case SingleInput::BUTTON_A:
            return SDL_CONTROLLER_BUTTON_A;
        case SingleInput::BUTTON_B:
            return SDL_CONTROLLER_BUTTON_B;
        case SingleInput::BUTTON_X:
            return SDL_CONTROLLER_BUTTON_X;
        case SingleInput::BUTTON_Y:
            return SDL_CONTROLLER_BUTTON_Y;
        case SingleInput::BUTTON_BACK:
            return SDL_CONTROLLER_BUTTON_BACK;
        case SingleInput::BUTTON_GUIDE:
            return SDL_CONTROLLER_BUTTON_GUIDE;
        case SingleInput::BUTTON_START:
            return SDL_CONTROLLER_BUTTON_START;
        case SingleInput::BUTTON_LEFTSTICK:
            return SDL_CONTROLLER_BUTTON_LEFTSTICK;
        case SingleInput::BUTTON_RIGHTSTICK:
            return SDL_CONTROLLER_BUTTON_RIGHTSTICK;
        case SingleInput::BUTTON_LEFTSHOULDER:
            return SDL_CONTROLLER_BUTTON_LEFTSHOULDER;
        case SingleInput::BUTTON_RIGHTSHOULDER:
            return SDL_CONTROLLER_BUTTON_RIGHTSHOULDER;
        case SingleInput::BUTTON_DPAD_UP:
            return SDL_CONTROLLER_BUTTON_DPAD_UP;
        case SingleInput::BUTTON_DPAD_DOWN:
            return SDL_CONTROLLER_BUTTON_DPAD_DOWN;
        case SingleInput::BUTTON_DPAD_LEFT:
            return SDL_CONTROLLER_BUTTON_DPAD_LEFT;
        case SingleInput::BUTTON_DPAD_RIGHT:
            return SDL_CONTROLLER_BUTTON_DPAD_RIGHT;
        default:
            return SDL_CONTROLLER_BUTTON_INVALID;
    }
}

int SingleInput::toSDLHat(int buttons)
{
    /* Fortunately, we use the fact that SDL_HAT_X constants
     * are the same in SDL 1 and SDL 2
     */
    Uint8 hatState = SDL_HAT_CENTERED;
    if (buttons & (1 << BUTTON_DPAD_UP))
        hatState |= SDL_HAT_UP;
    if (buttons & (1 << BUTTON_DPAD_DOWN))
        hatState |= SDL_HAT_DOWN;
    if (buttons & (1 << BUTTON_DPAD_LEFT))
        hatState |= SDL_HAT_LEFT;
    if (buttons & (1 << BUTTON_DPAD_RIGHT))
        hatState |= SDL_HAT_RIGHT;

    return hatState;
}


int SingleInput::toJsdevAxis(int axis)
{
    /* Mapping between xbox360 controller and joydev is taken from
     * http://wiki.unity3d.com/index.php/Xbox360Controller
     */
    switch(axis) {
        case SingleInput::AXIS_LEFTX:
            return 0;
        case SingleInput::AXIS_LEFTY:
            return 1;
        case SingleInput::AXIS_RIGHTX:
            return 3;
        case SingleInput::AXIS_RIGHTY:
            return 4;
        case SingleInput::AXIS_TRIGGERLEFT:
            return 2;
        case SingleInput::AXIS_TRIGGERRIGHT:
            return 5;
        default:
            return -1;
    }
}

int SingleInput::toJsdevButton(int button)
{
    /* Mapping between xbox360 controller and joydev is taken from
     * http://wiki.unity3d.com/index.php/Xbox360Controller
     */
    switch(button) {
        case SingleInput::BUTTON_A:
            return 0;
        case SingleInput::BUTTON_B:
            return 1;
        case SingleInput::BUTTON_X:
            return 2;
        case SingleInput::BUTTON_Y:
            return 3;
        case SingleInput::BUTTON_BACK:
            return 6;
        case SingleInput::BUTTON_GUIDE:
            return 8;
        case SingleInput::BUTTON_START:
            return 7;
        case SingleInput::BUTTON_LEFTSTICK:
            return 9;
        case SingleInput::BUTTON_RIGHTSTICK:
            return 10;
        case SingleInput::BUTTON_LEFTSHOULDER:
            return 4;
        case SingleInput::BUTTON_RIGHTSHOULDER:
            return 5;
        default:
            return -1;
    }
}

int SingleInput::toEvdevAxis(int axis)
{
    switch(axis) {
        case SingleInput::AXIS_LEFTX:
            return ABS_X;
        case SingleInput::AXIS_LEFTY:
            return ABS_Y;
        case SingleInput::AXIS_RIGHTX:
            return ABS_RX;
        case SingleInput::AXIS_RIGHTY:
            return ABS_RY;
        case SingleInput::AXIS_TRIGGERLEFT:
            return ABS_Z;
        case SingleInput::AXIS_TRIGGERRIGHT:
            return ABS_RZ;
        default:
            return -1;
    }
}

int SingleInput::toEvdevButton(int button)
{
    switch(button) {
        case SingleInput::BUTTON_A:
            return BTN_A;
        case SingleInput::BUTTON_B:
            return BTN_B;
        case SingleInput::BUTTON_X:
            return BTN_X;
        case SingleInput::BUTTON_Y:
            return BTN_Y;
        case SingleInput::BUTTON_BACK:
            return BTN_SELECT;
        case SingleInput::BUTTON_GUIDE:
            return BTN_MODE;
        case SingleInput::BUTTON_START:
            return BTN_START;
        case SingleInput::BUTTON_LEFTSTICK:
            return BTN_THUMBL;
        case SingleInput::BUTTON_RIGHTSTICK:
            return BTN_THUMBR;
        case SingleInput::BUTTON_LEFTSHOULDER:
            return BTN_TL;
        case SingleInput::BUTTON_RIGHTSHOULDER:
            return BTN_TR;
        default:
            return -1;
    }
}

int SingleInput::toDevHatX(int buttons)
{
    int hatx = 0;
    if (buttons & (1 << BUTTON_DPAD_LEFT))
        hatx = -1;
    else if (buttons & (1 << BUTTON_DPAD_RIGHT))
        hatx = 1;
    return hatx;
}

int SingleInput::toDevHatY(int buttons)
{
    int haty = 0;
    if (buttons & (1 << BUTTON_DPAD_UP))
        haty = -1;
    else if (buttons & (1 << BUTTON_DPAD_DOWN))
        haty = 1;
    return haty;
}
