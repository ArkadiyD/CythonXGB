#include "json.hpp"
#include <cstdint>
#include <cmath>
#include <cctype>
#include <string>
#include <deque>
#include <map>
#include <type_traits>
#include <initializer_list>
#include <ostream>
#include <iostream>
using namespace std;
using json::JSON;

JSON Array()
{
    return std::move( JSON::Make( JSON::Class::Array ) );
}

template <typename... T>
JSON Array( T... args )
{
    JSON arr = JSON::Make( JSON::Class::Array );
    arr.append( args... );
    return std::move( arr );
}

JSON Object()
{
    return std::move( JSON::Make( JSON::Class::Object ) );
}

std::ostream& operator<<( std::ostream &os, const JSON &json )
{
    os << json.dump();
    return os;
}