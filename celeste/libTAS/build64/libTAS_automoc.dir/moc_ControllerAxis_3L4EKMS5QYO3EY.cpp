/****************************************************************************
** Meta object code from reading C++ file 'ControllerAxisWidget.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.7.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../src/program/ui/ControllerAxisWidget.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'ControllerAxisWidget.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.7.1. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
struct qt_meta_stringdata_ControllerAxisWidget_t {
    QByteArrayData data[9];
    char stringdata0[90];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_ControllerAxisWidget_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_ControllerAxisWidget_t qt_meta_stringdata_ControllerAxisWidget = {
    {
QT_MOC_LITERAL(0, 0, 20), // "ControllerAxisWidget"
QT_MOC_LITERAL(1, 21, 12), // "XAxisChanged"
QT_MOC_LITERAL(2, 34, 0), // ""
QT_MOC_LITERAL(3, 35, 1), // "x"
QT_MOC_LITERAL(4, 37, 12), // "YAxisChanged"
QT_MOC_LITERAL(5, 50, 1), // "y"
QT_MOC_LITERAL(6, 52, 11), // "slotSetAxes"
QT_MOC_LITERAL(7, 64, 12), // "slotSetXAxis"
QT_MOC_LITERAL(8, 77, 12) // "slotSetYAxis"

    },
    "ControllerAxisWidget\0XAxisChanged\0\0x\0"
    "YAxisChanged\0y\0slotSetAxes\0slotSetXAxis\0"
    "slotSetYAxis"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_ControllerAxisWidget[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       5,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       2,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    1,   39,    2, 0x06 /* Public */,
       4,    1,   42,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
       6,    2,   45,    2, 0x0a /* Public */,
       7,    1,   50,    2, 0x0a /* Public */,
       8,    1,   53,    2, 0x0a /* Public */,

 // signals: parameters
    QMetaType::Void, QMetaType::Int,    3,
    QMetaType::Void, QMetaType::Int,    5,

 // slots: parameters
    QMetaType::Void, QMetaType::Short, QMetaType::Short,    3,    5,
    QMetaType::Void, QMetaType::Int,    3,
    QMetaType::Void, QMetaType::Int,    5,

       0        // eod
};

void ControllerAxisWidget::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        ControllerAxisWidget *_t = static_cast<ControllerAxisWidget *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->XAxisChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 1: _t->YAxisChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 2: _t->slotSetAxes((*reinterpret_cast< short(*)>(_a[1])),(*reinterpret_cast< short(*)>(_a[2]))); break;
        case 3: _t->slotSetXAxis((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 4: _t->slotSetYAxis((*reinterpret_cast< int(*)>(_a[1]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        void **func = reinterpret_cast<void **>(_a[1]);
        {
            typedef void (ControllerAxisWidget::*_t)(int );
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&ControllerAxisWidget::XAxisChanged)) {
                *result = 0;
                return;
            }
        }
        {
            typedef void (ControllerAxisWidget::*_t)(int );
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&ControllerAxisWidget::YAxisChanged)) {
                *result = 1;
                return;
            }
        }
    }
}

const QMetaObject ControllerAxisWidget::staticMetaObject = {
    { &QWidget::staticMetaObject, qt_meta_stringdata_ControllerAxisWidget.data,
      qt_meta_data_ControllerAxisWidget,  qt_static_metacall, Q_NULLPTR, Q_NULLPTR}
};


const QMetaObject *ControllerAxisWidget::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *ControllerAxisWidget::qt_metacast(const char *_clname)
{
    if (!_clname) return Q_NULLPTR;
    if (!strcmp(_clname, qt_meta_stringdata_ControllerAxisWidget.stringdata0))
        return static_cast<void*>(const_cast< ControllerAxisWidget*>(this));
    return QWidget::qt_metacast(_clname);
}

int ControllerAxisWidget::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
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

// SIGNAL 0
void ControllerAxisWidget::XAxisChanged(int _t1)
{
    void *_a[] = { Q_NULLPTR, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void ControllerAxisWidget::YAxisChanged(int _t1)
{
    void *_a[] = { Q_NULLPTR, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}
QT_END_MOC_NAMESPACE
