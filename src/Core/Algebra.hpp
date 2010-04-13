/****************************************************************************/
/* Copyright (c) 2009, Javor Kalojanov
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
* 
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*/
/****************************************************************************/

#ifdef _MSC_VER
#pragma once
#endif

#ifndef ALGEBRA_HPP_INCLUDED_4B8CA61D_C716_405F_9C54_A18D22E96CF1
#define ALGEBRA_HPP_INCLUDED_4B8CA61D_C716_405F_9C54_A18D22E96CF1

#include "../CUDAStdAfx.h"

#pragma region Structures on 4 components (vec4f and vec4i)

#define _DEF_BIN_OP4(_OP)                                                      \
    const t_this operator _OP (const t_this &aVec) const                       \
    {                                                                          \
    return t_this(x _OP aVec.x, y _OP aVec.y, z _OP aVec.z, w _OP aVec.w); \
    }                                                                          \
    t_this& operator _OP##= (const t_this &aVec)                               \
    {                                                                          \
    x _OP##= aVec.x; y _OP##= aVec.y; z _OP##= aVec.z; w _OP##= aVec.w;    \
    return *this;                                                          \
    }

#define _DEF_UNARY_MINUS4                                                      \
    const t_this operator- () const                                            \
    {                                                                          \
    return t_this(-x, -y, -z, -w);                                         \
    }                                                                          \

#define _DEF_CONSTR_AND_ACCESSORS4(_NAME)                                      \
    t_scalar x, y, z, w;	                                                   \
    _NAME() {}                                                                 \
    _NAME(t_scalar _x, t_scalar _y,                                            \
    t_scalar _z, t_scalar _w)                                              \
    : x(_x), y(_y), z(_z), w(_w)                                           \
        {}                                                                     \
        t_scalar& operator[] (int _index)                                          \
    {                                                                          \
    return (reinterpret_cast<t_scalar*>(this))[_index];                    \
    }                                                                          \
    const t_scalar& operator[] (int _index) const                              \
    {                                                                          \
    return (reinterpret_cast<const t_scalar*>(this))[_index];              \
    }                                                                          \
    template<int _i1, int _i2, int _i3, int _i4>                               \
    const t_this shuffle() const                                               \
    {                                                                          \
    return t_this((*this)[_i1], (*this)[_i2], (*this)[_i3], (*this)[_i4]); \
    }

#define _DEF_REP4                                                              \
    static t_this rep(t_scalar aVec) { return t_this(aVec, aVec, aVec, aVec); }

#define _DEF_CMPOP(_OP)                                                        \
    const t_cmpResult operator _OP (const t_this& _val) const                  \
    {                                                                          \
    return t_cmpResult(                                                    \
    x _OP _val.x ? (t_cmpResult::t_scalar)-1 : 0,                      \
    y _OP _val.y ? (t_cmpResult::t_scalar)-1 : 0,                      \
    z _OP _val.z ? (t_cmpResult::t_scalar)-1 : 0,                      \
    w _OP _val.w ? (t_cmpResult::t_scalar)-1 : 0                       \
    );                                                                     \
    }

#define _DEF_LOGOP(_OP, _TARG)                                                 \
    t_this& operator _OP##= (const _TARG &aVec)                                    \
{                                                                              \
    *(uint *)&x _OP##= *(uint*)&aVec.x;                                        \
    *(uint *)&y _OP##= *(uint*)&aVec.y;                                        \
    *(uint *)&z _OP##= *(uint*)&aVec.z;                                        \
    *(uint *)&w _OP##= *(uint*)&aVec.w;                                        \
    \
    return *this;                                                              \
}                                                                              \
    const t_this operator _OP (const _TARG &aVec) const                            \
{                                                                              \
    t_this ret = *this;                                                        \
    return ret _OP##= aVec;                                                    \
}

#define _DEF_LOGOP_SYM(_OP, _TARG)	  										   \
    friend t_this operator _OP (const _TARG &aVec, const t_this &_t)               \
{                                                                              \
    t_this ret = _t;                                                           \
    return ret _OP##= aVec;                                                    \
}


//A 4 component int32 value
struct vec4i
{
    typedef vec4i t_this;
    typedef int t_scalar;
    typedef vec4i t_cmpResult;

    //Constructors: default and vec4i(x, y, z, w)
    //Can also access the object using the [] operator
    _DEF_CONSTR_AND_ACCESSORS4(vec4i)

        //Bitwise operations with results from comparison. Very convenient for implementing
        //	per-component conditionals in the form (a _OP_ b ? a : b).
        _DEF_LOGOP(&, vec4i)
        _DEF_LOGOP(|, vec4i)
        _DEF_LOGOP(^, vec4i)

        const t_this operator~() const
    {
        vec4i ret = *this;
        ret.x = ~ret.x;
        ret.y = ~ret.y;
        ret.z = ~ret.z;
        ret.w = ~ret.w;

        return ret;
    }

    //Returns a mask containing the sign bit of each component
    int getMask() const
    {
        return (((uint)x >> 31) << 3) | (((uint)y >> 31) << 2) | (((uint)z >> 31) << 1) | ((uint)w >> 31);
    }
};

//The vec4f class can be used for color and vector calculations
struct vec4f
{
    typedef vec4f t_this;
    typedef float t_scalar;
    typedef vec4i t_cmpResult;

    //Constructors: default and vec4f(x, y, z, w)
    //Can also access the object using the [] operator
    _DEF_CONSTR_AND_ACCESSORS4(vec4f)

        //vec4f supports per-component +, -, * and /. Thus vec4f(1, 2, 3, 4) * vec4f(2, 2, 2, 2) gives vec4f(2, 4, 6, 8)
        _DEF_BIN_OP4(+)
        _DEF_BIN_OP4(-)
        _DEF_BIN_OP4(*)
        _DEF_BIN_OP4(/)

        //Per-component comparison operations. Return vec4i. For each component, the return value
        //	is 0 if the condition holds and -1 otherwise. You can use the getMask to see the result
        //	in a more compact form
        _DEF_CMPOP(<)
        _DEF_CMPOP(<=)
        _DEF_CMPOP(>)
        _DEF_CMPOP(>=)
        _DEF_CMPOP(==)
        _DEF_CMPOP(!=)

        //Bitwise operations with results from comparison. Very convenient for implementing
        //	per-component conditionals in the form (a _OP_ b ? a : b).
        //_DEF_LOGOP(&, vec4i)
        //_DEF_LOGOP_SYM(&, vec4i)
        //_DEF_LOGOP(&, vec4f)

        //_DEF_LOGOP(|, vec4i)
        //_DEF_LOGOP_SYM(|, vec4i)
        //_DEF_LOGOP(|, vec4f)

        //_DEF_LOGOP(^, vec4i)
        //_DEF_LOGOP_SYM(^, vec4i)
        //_DEF_LOGOP(^, vec4f)


        //An unary minus
        _DEF_UNARY_MINUS4

        //A replication function. You can use vec4f::rep(4.f) as a shortcut to vec4f(4.f, 4.f, 4.f, 4.f)
        _DEF_REP4

        //4 component dot product 
        float dot(const vec4f& aVec) const
    {
        return x * aVec.x + y * aVec.y + z * aVec.z + w * aVec.w;
    }

    //Cross product on the first three components. The .w component of the result has no meaning
    const vec4f cross(const vec4f& aVec) const
    {
        return shuffle<1, 2, 0, 3>() * aVec.shuffle<2, 0, 1, 3>() 
            - shuffle<2, 0, 1, 3>() * aVec.shuffle<1, 2, 0, 3>();
    }

    //A component-wise minimum between two vec4fs
    static const vec4f min(const vec4f & aVec1, const vec4f & aVec2)
    {
        return vec4f(
            cudastd::min(aVec1.x, aVec2.x), cudastd::min(aVec1.y, aVec2.y), 
            cudastd::min(aVec1.z, aVec2.z), cudastd::min(aVec1.w, aVec2.w)
            );
    }

    //A component-wise maximum between two vec4fs
    static const vec4f max(const vec4f & aVec1, const vec4f & aVec2)
    {
        return vec4f(
            cudastd::max(aVec1.x, aVec2.x), cudastd::max(aVec1.y, aVec2.y), 
            cudastd::max(aVec1.z, aVec2.z), cudastd::max(aVec1.w, aVec2.w)
            );
    }

    //Length of the vector
    float len() const
    {
        return sqrtf(this->dot(*this));
    }


    //Cross product
    vec4f operator %(const vec4f& aVec) const
    {
        vec4f t = this->cross(aVec);
        return vec4f(t.x, t.y, t.z, 0.f);
    }

    //A normalized vector: V / |V|
    vec4f operator ~() const
    {
        return *this / vec4f::rep(len());
    }
};

#undef _DEF_BIN_OP4
#undef _DEF_UNARY_MINUS4
#undef _DEF_CONSTR_AND_ACCESSORS4
#undef _DEF_REP4
#undef _DEF_CMPOP
#undef _DEF_LOGOP
#undef _DEF_LOGOP_SYM

#pragma endregion

#pragma region Structures on 3 components (vec3i and vec3f)

#define _DEF_BIN_OP3(_OP)                                                      \
    const t_this operator _OP (const t_this &aVec) const                       \
{                                                                          \
    t_this retval;                                                         \
    retval.x = x _OP aVec.x;                                               \
    retval.y = y _OP aVec.y;                                               \
    retval.z = z _OP aVec.z;                                               \
    return retval;                                                         \
}                                                                          \
    t_this& operator _OP##= (const t_this &aVec)                               \
{                                                                          \
    x  = x _OP aVec.x; y = y _OP aVec.y; z = z _OP aVec.z;                     \
    return *this;                                                              \
}

#define _DEF_UNARY_MINUS3                                                      \
    const t_this operator- () const                                            \
{                                                                          \
    t_this retval;                                                         \
    retval.x = -x;                                                         \
    retval.y = -y;                                                         \
    retval.z = -z;                                                         \
    return retval;                                                         \
}                                                                          \

#define _DEF_SCALAR_OP3(_OP, _TARG)                                            \
    const t_this operator _OP (_TARG aVal) const                               \
{                                                                          \
    t_this retval;                                                         \
    retval.x = x _OP aVal;                                                 \
    retval.y = y _OP aVal;                                                 \
    retval.z = z _OP aVal;                                                 \
    return retval;                                                         \
}                                                                          \
    t_this& operator _OP##= (const _TARG _v)                                   \
{                                                                          \
    x _OP##= _v; y _OP##= _v; z _OP##= _v;                                 \
    return *this;                                                          \
}

#define _DEF_SCALAR_OP3_SYM(_OP, _TARG)                                        \
    friend const t_this operator _OP (_TARG aVal, t_this aVec)                 \
{                                                                          \
    t_this retval;                                                         \
    retval.x = aVal _OP aVec.x;                                            \
    retval.y = aVal _OP aVec.y;                                            \
    retval.z = aVal _OP aVec.z;                                            \
    return retval;                                                         \
}                                                                          \


#define _DEF_CONSTR_AND_ACCESSORS3(_NAME)                                      \
    t_scalar x, y, z;	                                                       \
    \
    t_scalar& operator[] (int _index)                                         \
{                                                                          \
    return (reinterpret_cast<t_scalar*>(this))[_index];                    \
}                                                                          \
    const t_scalar& operator[] (int _index) const                              \
{                                                                          \
    return (reinterpret_cast<const t_scalar*>(this))[_index];              \
}                                                                          \
    template<int _i1, int _i2, int _i3>                                        \
    const t_this shuffle() const                                               \
{                                                                          \
    t_this retval;                                                         \
    retval.x = (*this)[_i1];                                               \
    retval.y = (*this)[_i2];                                               \
    retval.z = (*this)[_i3];                                               \
    return retval;                                                         \
}

#define _DEF_REP3                                                              \
    static DEVICE HOST t_this rep(t_scalar aVal)                               \
{                                                                          \
    t_this retval;                                                         \
    retval.x = aVal;                                                       \
    retval.y = aVal;                                                       \
    retval.z = aVal;                                                       \
    return retval;                                                         \
}

#define _DEF_CMPOP(_OP)                                                        \
    const t_cmpResult operator _OP (const t_this& aVal) const                  \
{                                                                          \
    t_cmpResult retval;                                                    \
    retval.x = x _OP aVal.x ? (t_cmpResult::t_scalar)-1 : 0;               \
    retval.y = y _OP aVal.y ? (t_cmpResult::t_scalar)-1 : 0;               \
    retval.z = z _OP aVal.z ? (t_cmpResult::t_scalar)-1 : 0;               \
    return retval;                                                         \
}

#define _DEF_LOGOP(_OP, _TARG)                                                 \
    t_this& operator _OP##= (const _TARG &aVec)                                \
{                                                                          \
    *(uint *)&x _OP##= *(uint*)&aVec.x;                                    \
    *(uint *)&y _OP##= *(uint*)&aVec.y;                                    \
    *(uint *)&z _OP##= *(uint*)&aVec.z;                                    \
    \
    return *this;                                                          \
}                                                                          \
    const t_this operator _OP (const _TARG &aVec) const                    \
{                                                                          \
    t_this ret = *this;                                                    \
    return ret _OP##= aVec;                                                \
}

#define _DEF_LOGOP_SYM(_OP, _TARG)	  										   \
    friend t_this operator _OP (const _TARG &aVec, const t_this &_t)           \
{                                                                          \
    t_this ret = _t;                                                       \
    return ret _OP##= aVec;                                                \
}

//A 3 component int32 value
struct vec3i
{
    typedef vec3i t_this;
    typedef int t_scalar;
    typedef vec3i t_cmpResult;

    //Constructors: default and vec3i(x, y, z)
    //Can also access the object using the [] operator
    _DEF_CONSTR_AND_ACCESSORS3(vec3i)

        //Bitwise operations with results from comparison. Very convenient for implementing
        //	per-component conditionals in the form (a _OP_ b ? a : b).
        _DEF_LOGOP(&, vec3i)
        _DEF_LOGOP(|, vec3i)
        _DEF_LOGOP(^, vec3i)

        const t_this operator~() const
    {
        vec3i ret = *this;
        ret.x = ~ret.x;
        ret.y = ~ret.y;
        ret.z = ~ret.z;

        return ret;
    }

    //Returns a mask containing the sign bit of each component
    int getMask() const
    {
        return ((((uint)x >> 31) << 2) | (((uint)y >> 31) << 1) | ((uint)z >> 31) );
    }
};

//The vec3f class can be used for color and vector calculations
struct vec3f
{
    typedef vec3f t_this;
    typedef float t_scalar;
    typedef vec3i t_cmpResult;

    //Constructors: default and vec3f(x, y, z)
    //Can also access the object using the [] operator
    _DEF_CONSTR_AND_ACCESSORS3(vec3f)

        //vec3f supports per-component +, -, * and /. Thus vec3f(1, 2, 3) * vec3f(2, 2, 2) gives vec3f(2, 4, 6)
        _DEF_BIN_OP3(+)
        _DEF_BIN_OP3(-)
        _DEF_BIN_OP3(*)
        _DEF_BIN_OP3(/)

        _DEF_SCALAR_OP3(*, float) //Vector * scalar -> Vector and Vector *= scalar
        _DEF_SCALAR_OP3_SYM(*, float) //scalar * Vector -> Vector
        _DEF_SCALAR_OP3(/, float) //Vector / scalar and Vector /= scalar

        static DEVICE vec3f fastDivide(const vec3f & aVec1, const vec3f & aVec2)
    {
#ifdef __CUDACC__
        vec3f retval;
        retval.x = __fdividef(aVec1.x, aVec2.x);
        retval.y = __fdividef(aVec1.y, aVec2.y); 
        retval.z = __fdividef(aVec1.z, aVec2.z);
        return retval;
#else
        return aVec1 / aVec2;
#endif
    }

    static DEVICE vec3f fastDivide(const vec3f& aVec1, const float& aVal)
    {
#ifdef __CUDACC__
        vec3f retval;
        retval.x = __fdividef(aVec1.x, aVal);
        retval.y = __fdividef(aVec1.y, aVal); 
        retval.z = __fdividef(aVec1.z, aVal);
        return retval;
#else
        return aVec1 / aVal;
#endif
    }

    //Per-component comparison operations. Return vec4i. For each component, the return value
    //	is 0 if the condition holds and -1 otherwise. You can use the getMask to see the result
    //	in a more compact form
    _DEF_CMPOP(<)
    _DEF_CMPOP(<=)
    _DEF_CMPOP(>)
    _DEF_CMPOP(>=)
    _DEF_CMPOP(==)
    _DEF_CMPOP(!=)

    //Bitwise operations with results from comparison. Very convenient for implementing
    //	per-component conditionals in the form (a _OP_ b ? a : b).
    //_DEF_LOGOP(&, vec3i)
    //_DEF_LOGOP_SYM(&, vec3i)
    //_DEF_LOGOP(&, vec3f)

    //_DEF_LOGOP(|, vec3i)
    //_DEF_LOGOP_SYM(|, vec3i)
    //_DEF_LOGOP(|, vec3f)

    //_DEF_LOGOP(^, vec3i)
    //_DEF_LOGOP_SYM(^, vec3i)
    //_DEF_LOGOP(^, vec3f)


    //An unary minus
    _DEF_UNARY_MINUS3

    //A replication function. You can use vec3f::rep(4.f) as a shortcut to vec3f(4.f, 4.f, 4.f)
    _DEF_REP3

    //dot product 
    float DEVICE HOST dot(const vec3f& aVec) const
    {
        return x * aVec.x + y * aVec.y + z * aVec.z;
    }

    //dot product with the reciprocal of this
    float DEVICE dotRCP(const vec3f& aVec) const
    {
#ifdef __CUDACC__
        return  __fdividef(aVec.x, x) +
            __fdividef(aVec.y, y) +
            __fdividef(aVec.z, z);
#else
        return aVec.x / x + aVec.y / y + aVec.z / z;
#endif
    }

    //Cross product on the first three components
    const DEVICE HOST vec3f cross(const vec3f& aVec) const
    {
        vec3f retval;
        retval.x = y * aVec.z - z * aVec.y;
        retval.y = z * aVec.x - x * aVec.z;
        retval.z = x * aVec.y - y * aVec.x;
        return retval;
    }

    //Cross product with the reciprocal of this
    const DEVICE vec3f crossRCP(const vec3f& aVec) const
    {
        vec3f retval;
#ifdef __CUDACC__
        retval.x = __fdividef(aVec.z, y) - __fdividef(aVec.y, z);
        retval.y = __fdividef(aVec.x, z) - __fdividef(aVec.z, x); 
        retval.z = __fdividef(aVec.y, x) - __fdividef(aVec.x, y);
#else
        retval.x = aVec.z / y - aVec.y / z;
        retval.y = aVec.x / z - aVec.z / x;
        retval.z = aVec.y / x - aVec.x / y;
#endif
        return retval;
    }

    //A component-wise maximum between two vec3fs
    static DEVICE HOST vec3f min(const vec3f & aVec1, const vec3f & aVec2)
    {
#ifdef __CUDACC__
        vec3f retval;
        retval.x = fminf(aVec1.x, aVec2.x);
        retval.y = fminf(aVec1.y, aVec2.y); 
        retval.z = fminf(aVec1.z, aVec2.z);
        return retval;
#else
        vec3f retval;
        retval.x = cudastd::min(aVec1.x, aVec2.x);
        retval.y = cudastd::min(aVec1.y, aVec2.y); 
        retval.z = cudastd::min(aVec1.z, aVec2.z);
        return retval;
#endif
    }

    static DEVICE HOST vec3f max(const vec3f & aVec1, const vec3f & aVec2)
    {
#ifdef __CUDACC__
        vec3f retval;
        retval.x = fmaxf(aVec1.x, aVec2.x);
        retval.y = fmaxf(aVec1.y, aVec2.y); 
        retval.z = fmaxf(aVec1.z, aVec2.z);
        return retval;
#else
        vec3f retval;
        retval.x = cudastd::max(aVec1.x, aVec2.x);
        retval.y = cudastd::max(aVec1.y, aVec2.y); 
        retval.z = cudastd::max(aVec1.z, aVec2.z);
        return retval;
#endif
    }


    //Length of the vector
    DEVICE HOST float len() const
    {
        return sqrtf(this->dot(*this));
    }

    DEVICE HOST float lenRCP() const
    {
#ifdef __CUDACC__
        return rsqrtf(this->dot(*this));
#else
        return 1.f / sqrtf(this->dot(*this));
#endif
    }

    //computes orthogonal local coordinate system
    static DEVICE HOST void getLocalCoordinates(
        const vec3f& aNormal,
        vec3f& oTangent,
        vec3f& oBinormal)
    {
        const int cId0  = (abs(aNormal.x) > abs(aNormal.y)) ? 0 : 1;
        const int cId1  = (abs(aNormal.x) > abs(aNormal.y)) ? 1 : 0;
        const float sig = (abs(aNormal.x) > abs(aNormal.y)) ? -1.f : 1.f;

        const float invLen = 1.f / (aNormal[cId0] * aNormal[cId0] +
            aNormal.z * aNormal.z);

        oTangent[cId0] = aNormal.z * sig * invLen;
        oTangent[cId1] = 0.f;
        oTangent.z   = aNormal[cId0] * -1.f * sig * invLen;

        oBinormal = aNormal.cross(oTangent);
    }

    //Cross product
    vec3f operator %(const vec3f& aVec) const
    {
        return this->cross(aVec);
    }

    //A normalized vector: V / |V|
    vec3f operator ~() const
    {
        return *this * vec3f::rep(lenRCP());
    }
};

#undef _DEF_BIN_OP3
#undef _DEF_UNARY_MINUS3
#undef _DEF_SCALAR_OP3
#undef _DEF_SCALAR_OP3_SYM
#undef _DEF_CONSTR_AND_ACCESSORS3
#undef _DEF_REP3
#undef _DEF_CMPOP
#undef _DEF_LOGOP
#undef _DEF_LOGOP_SYM

#pragma endregion

#pragma region Structures on 2 components (vec2f)

#define _DEF_SCALAR_OP2(_OP, _TARG)                                            \
    const t_this operator _OP (_TARG aVec) const                               \
{                                                                              \
    return t_this(x _OP aVec, y _OP aVec);                                     \
}                                                                              \
    t_this& operator _OP##= (const _TARG aVec)                                 \
{                                                                              \
    x _OP##= aVec; y _OP##= aVec;                                              \
    return *this;                                                              \
}

#define _DEF_SCALAR_OP2_SYM(_OP, _TARG)                                        \
    friend const t_this operator _OP (_TARG aVec1, t_this aVec2)               \
{                                                                              \
    return                                                                     \
    t_this(aVec1 _OP aVec2.x, aVec1 _OP aVec2.y);                              \
}                                                                              \

#define _DEF_BIN_OP2(_OP, _T_ARG)                                              \
    const t_this operator _OP (const _T_ARG &aVec) const                       \
{                                                                              \
    return t_this(x _OP aVec.x, y _OP aVec.y);                                 \
}                                                                              \
    \
    t_this& operator _OP##= (const _T_ARG &aVec)                               \
{                                                                              \
    x _OP##= aVec.x; y _OP##= aVec.y;                                          \
    return *this;                                                              \
}

#define _DEF_BIN_OP2_SYM(_OP, _T_ARG)                                          \
    friend const t_this operator _OP (const _T_ARG &aVec1,                     \
    const t_this &aVec2)                                                       \
{                                                                              \
    return                                                                     \
    t_this(aVec1.x _OP aVec2.x, aVec1.y _OP aVec2.y);                          \
}                                                                          

struct vec2f
{
    float x, y;
    vec2f(){}
    vec2f(float _x, float _y) : x(_x), y(_y) {}

    typedef vec2f t_this;
    typedef float t_scalar;

    _DEF_BIN_OP2(+, vec2f)
        _DEF_BIN_OP2(-, vec2f)
        _DEF_BIN_OP2(*, vec2f)
        _DEF_BIN_OP2(/, vec2f)

        _DEF_SCALAR_OP2(*, float)
        _DEF_SCALAR_OP2_SYM(*, float)
        _DEF_SCALAR_OP2(/, float)
        _DEF_SCALAR_OP2_SYM(/, float)
};

#undef _DEF_SCALAR_OP2
#undef _DEF_SCALAR_OP2_SYM
#undef _DEF_BIN_OP2
#undef _DEF_BIN_OP2_SYM

#pragma endregion

#pragma region Flag Container Structures

struct flag4
{
    //1st, 2nd, 3rd byte: data (uint)
    //4th byte: 4 boolean flags
    int data; //use int not uint or bool!

    flag4():data(0)
    {}

    enum{
        FLAG1_SHIFT     =   0,
        FLAG2_SHIFT     =   1,
        FLAG1_MASK      =   0x1,
        FLAG3_SHIFT     =   2,
        FLAG2_MASK      =   0x2,
        FLAG4_SHIFT     =   3,
        FLAG3_MASK      =   0x4,
        DATA_SHIFT      =   4,
        FLAG4_MASK      =   0x8,
        DATA_MASK       =   0xF,
        DATA_MASKNEG    =   0xFFFFFFF0,
        FLAG4_MASKNEG   =   0xFFFFFFF7,
        FLAG3_MASKNEG   =   0xFFFFFFFB,
        FLAG2_MASKNEG   =   0xFFFFFFFD,
        FLAG1_MASKNEG   =   0xFFFFFFFE,
    };

#define SET_FLAG(aFlagId)                                                      \
    void setFlag##aFlagId(bool aVal)                                           \
    { data = aVal ?                                                            \
    (data | FLAG##aFlagId##_MASK) : (data & FLAG##aFlagId##_MASKNEG); }

#define GET_FLAG(aFlagId)                                                      \
    bool getFlag##aFlagId () const                                             \
    { return (data & FLAG##aFlagId##_MASK) != 0x0; }

#define SET_FLAG_0(aFlagId)                                                    \
    void setFlag##aFlagId##To0 () { data &= FLAG##aFlagId##_MASKNEG; }

    SET_FLAG(1)
    SET_FLAG(2)
    SET_FLAG(3)
    SET_FLAG(4)

    GET_FLAG(1)
    GET_FLAG(2)
    GET_FLAG(3)
    GET_FLAG(4)

    SET_FLAG_0(1)
    SET_FLAG_0(2)
    SET_FLAG_0(3)
    SET_FLAG_0(4)

#undef SET_FLAG
#undef GET_FLAG
#undef SET_FLAG_0


    bool anyFlag() const { return (data & DATA_MASK) != 0x0; }
    bool noFlag() const { return !anyFlag(); }

    void setFlags12(bool aVal)
    {
        data = aVal ? data | FLAG1_MASK | FLAG2_MASK : 
            (data & FLAG1_MASKNEG) & FLAG2_MASKNEG;
    }

    void setData(const int aData){ data |= aData << DATA_SHIFT; }
    uint getData() const { return data >> DATA_SHIFT; }

};

#pragma endregion // Flag Container Structures

#endif // ALGEBRA_HPP_INCLUDED_4B8CA61D_C716_405F_9C54_A18D22E96CF1
