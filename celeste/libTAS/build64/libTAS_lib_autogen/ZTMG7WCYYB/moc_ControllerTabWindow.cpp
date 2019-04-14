/****************************************************************************
** Meta object code from reading C++ file 'ControllerTabWindow.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.7.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../src/program/ui/ControllerTabWindow.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'ControllerTabWindow.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.7.1. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
struct qt_meta_stringdata_ControllerTabWindow_t {
    QByteArrayData data[9];
    char stringdata0[84];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_ControllerTabWindow_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_ControllerTabWindow_t qt_meta_stringdata_ControllerTabWindow = {
    {
QT_MOC_LITERAL(0, 0, 19), // "ControllerTabWindow"
QT_MOC_LITERAL(1, 20, 16), // "slotButtonToggle"
QT_MOC_LITERAL(2, 37, 0), // ""
QT_MOC_LITERAL(3, 38, 2), // "id"
QT_MOC_LITERAL(4, 41, 6), // "button"
QT_MOC_LITERAL(5, 48, 7), // "pressed"
QT_MOC_LITERAL(6, 56, 13), // "slotSetInputs"
QT_MOC_LITERAL(7, 70, 10), // "AllInputs&"
QT_MOC_LITERAL(8, 81, 2) // "ai"

    },
    "ControllerTabWindow\0slotButtonToggle\0"
    "\0id\0button\0pressed\0slotSetInputs\0"
    "AllInputs&\0ai"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_ControllerTabWindow[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       2,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    3,   24,    2, 0x0a /* Public */,
       6,    1,   31,    2, 0x0a /* Public */,

 // slots: parameters
    QMetaType::Void, QMetaType::Int, QMetaType::Int, QMetaType::Bool,    3,    4,    5,
    QMetaType::Void, 0x80000000 | 7,    8,

       0        // eod
};

void ControllerTabWindow::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        ControllerTabWindow *_t = static_cast<ControllerTabWindow *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->slotButtonToggle((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2])),(*reinterpret_cast< bool(*)>(_a[3]))); break;
        case 1: _t->slotSetInputs((*reinterpret_cast< AllInputs(*)>(_a[1]))); break;
        default: ;
        }
    }
}

const QMetaObject ControllerTabWindow::staticMetaObject = {
    { &QDialog::staticMetaObject, qt_meta_stringdata_ControllerTabWindow.data,
      qt_meta_data_ControllerTabWindow,  qt_static_metacall, Q_NULLPTR, Q_NULLPTR}
};


const QMetaObject *ControllerTabWindow::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *ControllerTabWindow::qt_metacast(const char *_clname)
{
    if (!_clname) return Q_NULLPTR;
    if (!strcmp(_clname, qt_meta_stringdata_ControllerTabWindow.stringdata0))
        return static_cast<void*>(const_cast< ControllerTabWindow*>(this));
    return QDialog::qt_metacast(_clname);
}

int ControllerTabWindow::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QDialog::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 2)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 2;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 2)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 2;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
