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

#ifndef LIBTAS_RAMWATCHEDITWINDOW_H_INCLUDED
#define LIBTAS_RAMWATCHEDITWINDOW_H_INCLUDED

#include <QDialog>
#include <QLineEdit>
#include <QComboBox>
#include <QFormLayout>
#include <QVBoxLayout>
#include <QCheckBox>
#include <QDialogButtonBox>
#include <memory> // std::unique_ptr

#include "../ramsearch/IRamWatch.h"
#include "../ramsearch/IRamWatchDetailed.h"

class RamWatchEditWindow : public QDialog {
    Q_OBJECT

public:
    RamWatchEditWindow(QWidget *parent = Q_NULLPTR, Qt::WindowFlags flags = 0);

    void clear();
    void fill(std::unique_ptr<IRamWatchDetailed> &watch);
    void fill(std::unique_ptr<IRamWatch> &watch);

    std::unique_ptr<IRamWatchDetailed> ramwatch;

private:
    QLineEdit *addressInput;
    QLineEdit *valueInput;
    QLineEdit *labelInput;

    QComboBox *typeBox;
    QComboBox *displayBox;

    QCheckBox *pointerBox;

    // std::vector<std::unique_ptr<QSpinBox>> offsetBoxes;
    QVBoxLayout *mainLayout;
    QFormLayout *pointerLayout;
    QDialogButtonBox *buttonOffsetBox;
    QLineEdit *baseAddressInput;

private slots:
    void slotSave();
    void slotPoke();
    void slotPointer(bool checked);
    void slotAddOffset();
    void slotRemoveOffset();
};

#endif
