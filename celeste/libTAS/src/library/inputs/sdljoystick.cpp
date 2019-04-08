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

#include "sdljoystick.h"
#include "inputs.h"
#include "../logging.h"
#include "../sdlversion.h"
#include "../SDLEventQueue.h"
#include "../../shared/AllInputs.h"
#include "../../shared/SharedConfig.h"
#include <stdlib.h>

namespace libtas {

/* Override */ int SDL_NumJoysticks(void)
{
    debuglog(LCF_SDL | LCF_JOYSTICK, __func__, " call.");
    /* For now, we declare one joystick */
    return shared_config.nb_controllers;
}

const char* joyname = "Microsoft X-Box 360 pad";

/* Override */ const char *SDL_JoystickNameForIndex(int device_index)
{
    DEBUGLOGCALL(LCF_SDL | LCF_JOYSTICK);
    if (device_index < shared_config.nb_controllers)
        return joyname;
    return NULL;
}

/* Override */ const char* SDL_JoystickName(SDL_Joystick* joystick)
{
    DEBUGLOGCALL(LCF_SDL | LCF_JOYSTICK);
    /* Do not use joystick argument unless you know what you are doing.
     * Because SDL 1.2 can call this function, but the argument is
     * of type int.
     */
    return joyname;
}

#define MAX_SDLJOYS 4
static int joyids[MAX_SDLJOYS] = {-1, -1, -1, -1};
static int refids[4] = {0, 0, 0, 0}; // joystick open/close is ref-counted

/* Helper functions */
static bool isIdValid(SDL_Joystick* joy)
{
    if (joy == NULL)
        return false;
    int *joyid = reinterpret_cast<int*>(joy);
    if ((*joyid < 0) || (*joyid >= MAX_SDLJOYS) || (*joyid >= shared_config.nb_controllers))
        return false;
    return true;
}

static bool isIdValidOpen(SDL_Joystick* joy)
{
    if (!isIdValid(joy))
        return false;
    int *joyid = reinterpret_cast<int*>(joy);
    if (joyids[*joyid] == -1)
        return false;
    return true;
}

/* Override */ SDL_Joystick *SDL_JoystickOpen(int device_index)
{
    debuglog(LCF_SDL | LCF_JOYSTICK, __func__, " call with joy ", device_index);
    if ((device_index < 0) || (device_index >= MAX_SDLJOYS))
        return NULL;
    if (device_index >= shared_config.nb_controllers)
        return NULL;

    /* Opening the joystick device */
    joyids[device_index] = device_index;

    /* Increment ref count */
    refids[device_index]++;

    return reinterpret_cast<SDL_Joystick*>(&joyids[device_index]);
}

/* Override */ SDL_Joystick *SDL_JoystickFromInstanceID(SDL_JoystickID joyid)
{
    debuglog(LCF_SDL | LCF_JOYSTICK, __func__, " call with joy id ", joyid);

    if ((joyid < 0) || (joyid >= MAX_SDLJOYS))
        return NULL;
    if (joyid >= shared_config.nb_controllers)
        return NULL;
    if (joyids[joyid] == -1)
        /* Device not opened */
        return NULL;

    return reinterpret_cast<SDL_Joystick*>(&joyids[joyid]);
}

/* Override */ Uint16 SDL_JoystickGetDeviceVendor(int device_index)
{
    debuglog(LCF_SDL | LCF_JOYSTICK, __func__, " call with joy ", device_index);
    if ((device_index < 0) || (device_index >= MAX_SDLJOYS))
        return 0;
    if (device_index >= shared_config.nb_controllers)
        return 0;

    return 0x045e; // vendor of the wired xbox360 controller
}

/* Override */ Uint16 SDL_JoystickGetDeviceProduct(int device_index)
{
    debuglog(LCF_SDL | LCF_JOYSTICK, __func__, " call with joy ", device_index);
    if ((device_index < 0) || (device_index >= MAX_SDLJOYS))
        return 0;
    if (device_index >= shared_config.nb_controllers)
        return 0;

    return 0x028e; // product of the wired xbox360 controller
}

/* Override */ Uint16 SDL_JoystickGetDeviceProductVersion(int device_index)
{
    debuglog(LCF_SDL | LCF_JOYSTICK, __func__, " call with joy ", device_index);
    if ((device_index < 0) || (device_index >= MAX_SDLJOYS))
        return 0;
    if (device_index >= shared_config.nb_controllers)
        return 0;

    return 0x0114; // product version of the wired xbox360 controller
}

/* Override */ SDL_JoystickType SDL_JoystickGetDeviceType(int device_index)
{
    debuglog(LCF_SDL | LCF_JOYSTICK, __func__, " call with joy ", device_index);
    if ((device_index < 0) || (device_index >= MAX_SDLJOYS))
        return SDL_JOYSTICK_TYPE_UNKNOWN;
    if (device_index >= shared_config.nb_controllers)
        return SDL_JOYSTICK_TYPE_UNKNOWN;

    return SDL_JOYSTICK_TYPE_GAMECONTROLLER;
}

/* Xbox 360 GUID */
SDL_JoystickGUID xinputGUID = {{0x03,0x00,0x00,0x00,0x5e,0x04,0x00,0x00,0x8e,0x02,0x00,0x00,0x14,0x01,0x00,0x00}};
SDL_JoystickGUID nullGUID   = {{0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00}};

/* Override */ SDL_JoystickGUID SDL_JoystickGetDeviceGUID(int device_index)
{
    debuglog(LCF_SDL | LCF_JOYSTICK, __func__, " call with device ", device_index);
    if (device_index >= shared_config.nb_controllers)
	    return nullGUID;

    return xinputGUID;
}

/* Override */ Uint16 SDL_JoystickGetVendor(SDL_Joystick * joystick)
{
    debuglog(LCF_SDL | LCF_JOYSTICK, __func__, " call with joy ", joystick?*reinterpret_cast<int*>(joystick):-1);

    if (!isIdValid(joystick))
        return 0;

    return 0x045e; // vendor of the wired xbox360 controller
}

/* Override */ Uint16 SDL_JoystickGetProduct(SDL_Joystick * joystick)
{
    debuglog(LCF_SDL | LCF_JOYSTICK, __func__, " call with joy ", joystick?*reinterpret_cast<int*>(joystick):-1);

    if (!isIdValid(joystick))
        return 0;

    return 0x028e; // product of the wired xbox360 controller
}

/* Override */ Uint16 SDL_JoystickGetProductVersion(SDL_Joystick * joystick)
{
    debuglog(LCF_SDL | LCF_JOYSTICK, __func__, " call with joy ", joystick?*reinterpret_cast<int*>(joystick):-1);

    if (!isIdValid(joystick))
        return 0;

    return 0x0114; // product version of the wired xbox360 controller
}

/* Override */ SDL_JoystickType SDL_JoystickGetType(SDL_Joystick * joystick)
{
    debuglog(LCF_SDL | LCF_JOYSTICK, __func__, " call with joy ", joystick?*reinterpret_cast<int*>(joystick):-1);

    if (!isIdValid(joystick))
        return SDL_JOYSTICK_TYPE_UNKNOWN;

    return SDL_JOYSTICK_TYPE_GAMECONTROLLER;
}

/* Override */ SDL_JoystickGUID SDL_JoystickGetGUID(SDL_Joystick * joystick)
{
    DEBUGLOGCALL(LCF_SDL | LCF_JOYSTICK);
    if (!isIdValid(joystick))
        return nullGUID;

    return xinputGUID;
}

/* Override */ SDL_bool SDL_JoystickGetAttached(SDL_Joystick * joystick)
{
    debuglog(LCF_SDL | LCF_JOYSTICK, __func__, " call with joy ", joystick?*reinterpret_cast<int*>(joystick):-1);
    if (!isIdValidOpen(joystick))
        return SDL_FALSE;

    return SDL_TRUE;
}

/* Override */ int SDL_JoystickOpened(int device_index)
{
    debuglog(LCF_SDL | LCF_JOYSTICK, __func__, " call with joy ", device_index);
    if (!isIdValidOpen(reinterpret_cast<SDL_Joystick*>(&device_index)))
        return 0;

    return 1;
}

/* Override */ SDL_JoystickID SDL_JoystickInstanceID(SDL_Joystick * joystick)
{
    debuglog(LCF_SDL | LCF_JOYSTICK, __func__, " call with joy ", joystick?*reinterpret_cast<int*>(joystick):-1);
    if (!isIdValid(joystick))
        return -1;

    /* This function can be called without the joystick been opened... */
    return static_cast<SDL_JoystickID>(*reinterpret_cast<int*>(joystick));
}

int SDL_JoystickIndex(SDL_Joystick *joystick)
{
    debuglog(LCF_SDL | LCF_JOYSTICK, __func__, " call with joy ", joystick?*reinterpret_cast<int*>(joystick):-1);
    if (!isIdValidOpen(joystick))
        return -1;

    return *reinterpret_cast<int*>(joystick);
}

/* Override */ int SDL_JoystickNumAxes(SDL_Joystick* joystick)
{
    DEBUGLOGCALL(LCF_SDL | LCF_JOYSTICK);
    if (!isIdValid(joystick))
        return 0;
    return 6;
}

/* Override */ int SDL_JoystickNumBalls(SDL_Joystick* joystick)
{
    DEBUGLOGCALL(LCF_SDL | LCF_JOYSTICK);
    if (!isIdValid(joystick))
        return 0;
    return 0;
}

/* Override */ int SDL_JoystickNumButtons(SDL_Joystick* joystick)
{
    DEBUGLOGCALL(LCF_SDL | LCF_JOYSTICK);
    if (!isIdValid(joystick))
        return 0;
    return 11;
}

/* Override */ int SDL_JoystickNumHats(SDL_Joystick* joystick)
{
    DEBUGLOGCALL(LCF_SDL | LCF_JOYSTICK);
    if (!isIdValid(joystick))
        return 0;
    return 1;
}

/* Override */ void SDL_JoystickUpdate(void)
{
    DEBUGLOGCALL(LCF_SDL | LCF_JOYSTICK);

    for (int j=0; j<shared_config.nb_controllers; j++) {
        for (int a=0; a<AllInputs::MAXAXES; a++)
            game_ai.controller_axes[j][a] = ai.controller_axes[j][a];
        game_ai.controller_buttons[j] = ai.controller_buttons[j];
    }
}

/* Override */ int SDL_JoystickEventState(int state)
{
    debuglog(LCF_SDL | LCF_JOYSTICK, __func__, " call with state ", state);
    const int joyevents1[] = {
        SDL1::SDL_JOYAXISMOTION,
        SDL1::SDL_JOYBUTTONDOWN,
        SDL1::SDL_JOYBUTTONUP,
        SDL1::SDL_JOYHATMOTION,
        SDL1::SDL_JOYBALLMOTION
    };

    const int joyevents2[] = {
        SDL_JOYAXISMOTION,
        SDL_JOYBUTTONDOWN,
        SDL_JOYBUTTONUP,
        SDL_JOYHATMOTION,
        SDL_JOYBALLMOTION,
        SDL_JOYDEVICEADDED,
        SDL_JOYDEVICEREMOVED
    };

    bool enabled = false;
    int SDLver = get_sdlversion();

    switch (state) {
        case SDL_ENABLE:
            if (SDLver == 1)
                for (int e=0; e<5; e++)
                    sdlEventQueue.enable(joyevents1[e]);
            if (SDLver == 2)
                for (int e=0; e<7; e++)
                    sdlEventQueue.enable(joyevents2[e]);
            return 1;
        case SDL_IGNORE:
            if (SDLver == 1)
                for (int e=0; e<5; e++)
                    sdlEventQueue.disable(joyevents1[e]);
            if (SDLver == 2)
                for (int e=0; e<7; e++)
                    sdlEventQueue.disable(joyevents2[e]);
            return 0;
        case SDL_QUERY:
            if (SDLver == 1)
                for (int e=0; e<5; e++)
                    enabled = enabled || sdlEventQueue.isEnabled(joyevents1[e]);
            if (SDLver == 2)
                for (int e=0; e<7; e++)
                    enabled = enabled || sdlEventQueue.isEnabled(joyevents2[e]);
            if (enabled)
                return SDL_ENABLE;
            return SDL_IGNORE;
        default:
            return state;
    }
}

/* Override */ Sint16 SDL_JoystickGetAxis(SDL_Joystick * joystick, int axis)
{
    debuglog(LCF_SDL | LCF_JOYSTICK, __func__, " call with axis ", axis);

    if (!isIdValidOpen(joystick))
        return 0;

    if (axis >= 6)
        return 0;

    int *joyid = reinterpret_cast<int*>(joystick);

    return game_ai.controller_axes[*joyid][axis];
}

/* Override */ Uint8 SDL_JoystickGetHat(SDL_Joystick * joystick, int hat)
{
    debuglog(LCF_SDL | LCF_JOYSTICK, __func__, " call with hat ", hat);

    if (!isIdValidOpen(joystick))
        return 0;

    if (hat > 0)
        return 0;

    int *joyid = reinterpret_cast<int*>(joystick);

    Uint8 hatState = SDL_HAT_CENTERED;
    if (game_ai.controller_buttons[*joyid] & (1 << SDL_CONTROLLER_BUTTON_DPAD_UP))
        hatState |= SDL_HAT_UP;
    if (game_ai.controller_buttons[*joyid] & (1 << SDL_CONTROLLER_BUTTON_DPAD_DOWN))
        hatState |= SDL_HAT_DOWN;
    if (game_ai.controller_buttons[*joyid] & (1 << SDL_CONTROLLER_BUTTON_DPAD_LEFT))
        hatState |= SDL_HAT_LEFT;
    if (game_ai.controller_buttons[*joyid] & (1 << SDL_CONTROLLER_BUTTON_DPAD_RIGHT))
        hatState |= SDL_HAT_RIGHT;

    return hatState;
}

/* Override */ int SDL_JoystickGetBall(SDL_Joystick * joystick, int ball, int *dx, int *dy)
{
    debuglog(LCF_SDL | LCF_JOYSTICK, __func__, " call with ball ", ball);
    return 0;
}

/* Override */ Uint8 SDL_JoystickGetButton(SDL_Joystick * joystick, int button)
{
    debuglog(LCF_SDL | LCF_JOYSTICK, __func__, " call with button ", button);

    if (!isIdValidOpen(joystick))
        return 0;

    if (button >= 11)
        return 0;

    int *joyid = reinterpret_cast<int*>(joystick);

    return (game_ai.controller_buttons[*joyid] >> button) & 0x1;
}

/* Override */ void SDL_JoystickClose(SDL_Joystick * joystick)
{
    debuglog(LCF_SDL | LCF_JOYSTICK, __func__, " call with joy ", joystick?*reinterpret_cast<int*>(joystick):-1);
    if (!isIdValidOpen(joystick))
        return;

    int *joyid = reinterpret_cast<int*>(joystick);

    /* Decrease the ref count */
    refids[*joyid]--;

    /* If no more ref, close the joystick */
    if (refids[*joyid] == 0)
        joyids[*joyid] = -1;
}

/* Override */ SDL_JoystickPowerLevel SDL_JoystickCurrentPowerLevel(SDL_Joystick * joystick)
{
    DEBUGLOGCALL(LCF_SDL | LCF_JOYSTICK);
	return SDL_JOYSTICK_POWER_WIRED;
}

}
