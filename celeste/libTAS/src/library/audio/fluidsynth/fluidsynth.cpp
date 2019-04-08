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

#include "fluidsynth.h"
#include "../../logging.h"
#include "../../hook.h"

namespace libtas {

DEFINE_ORIG_POINTER(fluid_settings_getstr_default);
DEFINE_ORIG_POINTER(fluid_settings_setstr);
DEFINE_ORIG_POINTER(new_fluid_settings);
DEFINE_ORIG_POINTER(fluid_audio_driver_register);

static const char* alsa_driver = "alsa";

int fluid_settings_getstr_default(fluid_settings_t *settings, const char *name, char **def)
{
    debuglogstdio(LCF_SOUND, "%s called with name %s", __func__, name);

    if (strcmp(name, "audio.driver") == 0) {
        *def = const_cast<char*>(alsa_driver);
        return 0; // FLUID_OK
    }

    LINK_NAMESPACE(fluid_settings_getstr_default, "fluidsynth");
    return orig::fluid_settings_getstr_default(settings, name, def);
}

int fluid_settings_setstr(fluid_settings_t *settings, const char *name, const char *str)
{
    debuglogstdio(LCF_SOUND, "%s called with name %s", __func__, name);

    LINK_NAMESPACE(fluid_settings_setstr, "fluidsynth");

    int ret;

    if (strcmp(name, "audio.driver") == 0) {
        ret = orig::fluid_settings_setstr(settings, name, alsa_driver);
    }
    else {
        ret = orig::fluid_settings_setstr(settings, name, str);
    }

    return ret;
}

fluid_settings_t* new_fluid_settings(void)
{
    DEBUGLOGCALL(LCF_SOUND);

    /* Before creating the settings, we unregister every audio drivers except ALSA */
    LINK_NAMESPACE(fluid_audio_driver_register, "fluidsynth");
    const char* alsadriver = "alsa";
    const char* adrivers[2] = {alsadriver, nullptr};
    int ret = orig::fluid_audio_driver_register(adrivers);

    if (ret != 0) {
        debuglogstdio(LCF_SOUND | LCF_WARNING, "Could not register alsa driver");
    } // FLUID_OK

    /* Then return the original function */
    LINK_NAMESPACE(new_fluid_settings, "fluidsynth");
    return orig::new_fluid_settings();
}


}
