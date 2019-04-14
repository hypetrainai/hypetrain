/****************************************************************************
** Meta object code from reading C++ file 'InputEditorModel.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.7.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../src/program/ui/InputEditorModel.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'InputEditorModel.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.7.1. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
struct qt_meta_stringdata_InputEditorModel_t {
    QByteArrayData data[15];
    char stringdata0[191];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_InputEditorModel_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_InputEditorModel_t qt_meta_stringdata_InputEditorModel = {
    {
QT_MOC_LITERAL(0, 0, 16), // "InputEditorModel"
QT_MOC_LITERAL(1, 17, 17), // "frameCountChanged"
QT_MOC_LITERAL(2, 35, 0), // ""
QT_MOC_LITERAL(3, 36, 15), // "inputSetChanged"
QT_MOC_LITERAL(4, 52, 11), // "toggleInput"
QT_MOC_LITERAL(5, 64, 5), // "index"
QT_MOC_LITERAL(6, 70, 17), // "beginModifyInputs"
QT_MOC_LITERAL(7, 88, 15), // "endModifyInputs"
QT_MOC_LITERAL(8, 104, 14), // "beginAddInputs"
QT_MOC_LITERAL(9, 119, 12), // "endAddInputs"
QT_MOC_LITERAL(10, 132, 15), // "beginEditInputs"
QT_MOC_LITERAL(11, 148, 13), // "endEditInputs"
QT_MOC_LITERAL(12, 162, 17), // "registerSavestate"
QT_MOC_LITERAL(13, 180, 4), // "slot"
QT_MOC_LITERAL(14, 185, 5) // "frame"

    },
    "InputEditorModel\0frameCountChanged\0\0"
    "inputSetChanged\0toggleInput\0index\0"
    "beginModifyInputs\0endModifyInputs\0"
    "beginAddInputs\0endAddInputs\0beginEditInputs\0"
    "endEditInputs\0registerSavestate\0slot\0"
    "frame"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_InputEditorModel[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
      10,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       2,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    0,   64,    2, 0x06 /* Public */,
       3,    0,   65,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
       4,    1,   66,    2, 0x0a /* Public */,
       6,    0,   69,    2, 0x0a /* Public */,
       7,    0,   70,    2, 0x0a /* Public */,
       8,    0,   71,    2, 0x0a /* Public */,
       9,    0,   72,    2, 0x0a /* Public */,
      10,    0,   73,    2, 0x0a /* Public */,
      11,    0,   74,    2, 0x0a /* Public */,
      12,    2,   75,    2, 0x0a /* Public */,

 // signals: parameters
    QMetaType::Void,
    QMetaType::Void,

 // slots: parameters
    QMetaType::Bool, QMetaType::QModelIndex,    5,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int, QMetaType::ULong,   13,   14,

       0        // eod
};

void InputEditorModel::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        InputEditorModel *_t = static_cast<InputEditorModel *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->frameCountChanged(); break;
        case 1: _t->inputSetChanged(); break;
        case 2: { bool _r = _t->toggleInput((*reinterpret_cast< const QModelIndex(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< bool*>(_a[0]) = _r; }  break;
        case 3: _t->beginModifyInputs(); break;
        case 4: _t->endModifyInputs(); break;
        case 5: _t->beginAddInputs(); break;
        case 6: _t->endAddInputs(); break;
        case 7: _t->beginEditInputs(); break;
        case 8: _t->endEditInputs(); break;
        case 9: _t->registerSavestate((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< ulong(*)>(_a[2]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        void **func = reinterpret_cast<void **>(_a[1]);
        {
            typedef void (InputEditorModel::*_t)();
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&InputEditorModel::frameCountChanged)) {
                *result = 0;
                return;
            }
        }
        {
            typedef void (InputEditorModel::*_t)();
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&InputEditorModel::inputSetChanged)) {
                *result = 1;
                return;
            }
        }
    }
}

const QMetaObject InputEditorModel::staticMetaObject = {
    { &QAbstractTableModel::staticMetaObject, qt_meta_stringdata_InputEditorModel.data,
      qt_meta_data_InputEditorModel,  qt_static_metacall, Q_NULLPTR, Q_NULLPTR}
};


const QMetaObject *InputEditorModel::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *InputEditorModel::qt_metacast(const char *_clname)
{
    if (!_clname) return Q_NULLPTR;
    if (!strcmp(_clname, qt_meta_stringdata_InputEditorModel.stringdata0))
        return static_cast<void*>(const_cast< InputEditorModel*>(this));
    return QAbstractTableModel::qt_metacast(_clname);
}

int InputEditorModel::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QAbstractTableModel::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 10)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 10;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 10)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 10;
    }
    return _id;
}

// SIGNAL 0
void InputEditorModel::frameCountChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 0, Q_NULLPTR);
}

// SIGNAL 1
void InputEditorModel::inputSetChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 1, Q_NULLPTR);
}
QT_END_MOC_NAMESPACE
