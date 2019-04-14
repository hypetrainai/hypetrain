/****************************************************************************
** Meta object code from reading C++ file 'InputEditorView.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.7.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../src/program/ui/InputEditorView.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'InputEditorView.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.7.1. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
struct qt_meta_stringdata_InputEditorView_t {
    QByteArrayData data[20];
    char stringdata0[244];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_InputEditorView_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_InputEditorView_t qt_meta_stringdata_InputEditorView = {
    {
QT_MOC_LITERAL(0, 0, 15), // "InputEditorView"
QT_MOC_LITERAL(1, 16, 14), // "horizontalMenu"
QT_MOC_LITERAL(2, 31, 0), // ""
QT_MOC_LITERAL(3, 32, 3), // "pos"
QT_MOC_LITERAL(4, 36, 11), // "renameLabel"
QT_MOC_LITERAL(5, 48, 14), // "addInputColumn"
QT_MOC_LITERAL(6, 63, 16), // "clearInputColumn"
QT_MOC_LITERAL(7, 80, 15), // "lockInputColumn"
QT_MOC_LITERAL(8, 96, 7), // "checked"
QT_MOC_LITERAL(9, 104, 8), // "mainMenu"
QT_MOC_LITERAL(10, 113, 11), // "insertInput"
QT_MOC_LITERAL(11, 125, 12), // "insertInputs"
QT_MOC_LITERAL(12, 138, 11), // "deleteInput"
QT_MOC_LITERAL(13, 150, 14), // "truncateInputs"
QT_MOC_LITERAL(14, 165, 10), // "clearInput"
QT_MOC_LITERAL(15, 176, 10), // "copyInputs"
QT_MOC_LITERAL(16, 187, 9), // "cutInputs"
QT_MOC_LITERAL(17, 197, 11), // "pasteInputs"
QT_MOC_LITERAL(18, 209, 17), // "pasteInsertInputs"
QT_MOC_LITERAL(19, 227, 16) // "resizeAllColumns"

    },
    "InputEditorView\0horizontalMenu\0\0pos\0"
    "renameLabel\0addInputColumn\0clearInputColumn\0"
    "lockInputColumn\0checked\0mainMenu\0"
    "insertInput\0insertInputs\0deleteInput\0"
    "truncateInputs\0clearInput\0copyInputs\0"
    "cutInputs\0pasteInputs\0pasteInsertInputs\0"
    "resizeAllColumns"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_InputEditorView[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
      16,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    1,   94,    2, 0x0a /* Public */,
       4,    0,   97,    2, 0x0a /* Public */,
       5,    0,   98,    2, 0x0a /* Public */,
       6,    0,   99,    2, 0x0a /* Public */,
       7,    1,  100,    2, 0x0a /* Public */,
       9,    1,  103,    2, 0x0a /* Public */,
      10,    0,  106,    2, 0x0a /* Public */,
      11,    0,  107,    2, 0x0a /* Public */,
      12,    0,  108,    2, 0x0a /* Public */,
      13,    0,  109,    2, 0x0a /* Public */,
      14,    0,  110,    2, 0x0a /* Public */,
      15,    0,  111,    2, 0x0a /* Public */,
      16,    0,  112,    2, 0x0a /* Public */,
      17,    0,  113,    2, 0x0a /* Public */,
      18,    0,  114,    2, 0x0a /* Public */,
      19,    0,  115,    2, 0x08 /* Private */,

 // slots: parameters
    QMetaType::Void, QMetaType::QPoint,    3,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Bool,    8,
    QMetaType::Void, QMetaType::QPoint,    3,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,

       0        // eod
};

void InputEditorView::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        InputEditorView *_t = static_cast<InputEditorView *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->horizontalMenu((*reinterpret_cast< QPoint(*)>(_a[1]))); break;
        case 1: _t->renameLabel(); break;
        case 2: _t->addInputColumn(); break;
        case 3: _t->clearInputColumn(); break;
        case 4: _t->lockInputColumn((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 5: _t->mainMenu((*reinterpret_cast< QPoint(*)>(_a[1]))); break;
        case 6: _t->insertInput(); break;
        case 7: _t->insertInputs(); break;
        case 8: _t->deleteInput(); break;
        case 9: _t->truncateInputs(); break;
        case 10: _t->clearInput(); break;
        case 11: _t->copyInputs(); break;
        case 12: _t->cutInputs(); break;
        case 13: _t->pasteInputs(); break;
        case 14: _t->pasteInsertInputs(); break;
        case 15: _t->resizeAllColumns(); break;
        default: ;
        }
    }
}

const QMetaObject InputEditorView::staticMetaObject = {
    { &QTableView::staticMetaObject, qt_meta_stringdata_InputEditorView.data,
      qt_meta_data_InputEditorView,  qt_static_metacall, Q_NULLPTR, Q_NULLPTR}
};


const QMetaObject *InputEditorView::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *InputEditorView::qt_metacast(const char *_clname)
{
    if (!_clname) return Q_NULLPTR;
    if (!strcmp(_clname, qt_meta_stringdata_InputEditorView.stringdata0))
        return static_cast<void*>(const_cast< InputEditorView*>(this));
    return QTableView::qt_metacast(_clname);
}

int InputEditorView::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QTableView::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 16)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 16;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 16)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 16;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
