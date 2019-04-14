/****************************************************************************
** Meta object code from reading C++ file 'RamWatchWindow.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.7.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../src/program/ui/RamWatchWindow.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'RamWatchWindow.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.7.1. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
struct qt_meta_stringdata_RamWatchWindow_t {
    QByteArrayData data[9];
    char stringdata0[87];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_RamWatchWindow_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_RamWatchWindow_t qt_meta_stringdata_RamWatchWindow = {
    {
QT_MOC_LITERAL(0, 0, 14), // "RamWatchWindow"
QT_MOC_LITERAL(1, 15, 7), // "slotAdd"
QT_MOC_LITERAL(2, 23, 0), // ""
QT_MOC_LITERAL(3, 24, 7), // "slotGet"
QT_MOC_LITERAL(4, 32, 12), // "std::string&"
QT_MOC_LITERAL(5, 45, 5), // "watch"
QT_MOC_LITERAL(6, 51, 8), // "slotEdit"
QT_MOC_LITERAL(7, 60, 10), // "slotRemove"
QT_MOC_LITERAL(8, 71, 15) // "slotScanPointer"

    },
    "RamWatchWindow\0slotAdd\0\0slotGet\0"
    "std::string&\0watch\0slotEdit\0slotRemove\0"
    "slotScanPointer"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_RamWatchWindow[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       5,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    0,   39,    2, 0x0a /* Public */,
       3,    1,   40,    2, 0x0a /* Public */,
       6,    0,   43,    2, 0x08 /* Private */,
       7,    0,   44,    2, 0x08 /* Private */,
       8,    0,   45,    2, 0x08 /* Private */,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void, 0x80000000 | 4,    5,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,

       0        // eod
};

void RamWatchWindow::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        RamWatchWindow *_t = static_cast<RamWatchWindow *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->slotAdd(); break;
        case 1: _t->slotGet((*reinterpret_cast< std::string(*)>(_a[1]))); break;
        case 2: _t->slotEdit(); break;
        case 3: _t->slotRemove(); break;
        case 4: _t->slotScanPointer(); break;
        default: ;
        }
    }
}

const QMetaObject RamWatchWindow::staticMetaObject = {
    { &QDialog::staticMetaObject, qt_meta_stringdata_RamWatchWindow.data,
      qt_meta_data_RamWatchWindow,  qt_static_metacall, Q_NULLPTR, Q_NULLPTR}
};


const QMetaObject *RamWatchWindow::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *RamWatchWindow::qt_metacast(const char *_clname)
{
    if (!_clname) return Q_NULLPTR;
    if (!strcmp(_clname, qt_meta_stringdata_RamWatchWindow.stringdata0))
        return static_cast<void*>(const_cast< RamWatchWindow*>(this));
    return QDialog::qt_metacast(_clname);
}

int RamWatchWindow::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QDialog::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 5)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 5;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 5)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 5;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
