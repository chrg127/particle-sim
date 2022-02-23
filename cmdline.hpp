#pragma once

#include <cstdio>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace cmdline {

enum class ParamType { None, Single };

struct Argument {
    char short_opt;
    std::string_view long_opt;
    std::string_view desc;
    ParamType param_type = ParamType::None;
    std::string_view default_param = "";
};

using ArgumentList = std::vector<Argument>;

struct Result {
    std::unordered_map<char, bool> has;
    std::unordered_map<char, std::string_view> params;
    std::vector<std::string_view> items;
};

constexpr inline void warning(std::string_view fmtstr, auto&&... args)
{
    fmt::print(stderr, "warning: ");
    fmt::print(stderr, fmt::runtime(fmtstr), args...);
}

auto find_arg(std::string_view arg, const ArgumentList &list) { return std::find_if(list.begin(), list.end(), [&](const auto &a) { return a.long_opt  == arg; }); }
auto find_arg(char arg,             const ArgumentList &list) { return std::find_if(list.begin(), list.end(), [&](const auto &a) { return a.short_opt == arg; }); }

inline Result parse(const std::vector<std::string_view> &args, const ArgumentList &valid_args)
{
    Result res;
    for (auto it = args.begin()+1, it2 = args.begin()+2; it != args.end(); ++it, ++it2) {
        std::string_view curr = *it;

        if (curr[0] != '-') {
            res.items.push_back(curr);
            continue;
        }

        auto arg = curr[1] == '-'   ? find_arg(curr.substr(2), valid_args)
                 : curr.size() == 2 ? find_arg(curr[1], valid_args)
                 :                    valid_args.end();
        if (arg == valid_args.end()) {
            warning("invalid argument: {}\n", curr);
            continue;
        }
        if (res.has[arg->short_opt]) {
            warning("argument {} was specified multiple times\n", curr);
            continue;
        }
        res.has[arg->short_opt] = true;

        if (arg->param_type != ParamType::None) {
            ++it;
            if (it == args.end()) {
                warning("argument --{} needs a parameter (default \"{}\" will be used)\n", arg->long_opt, arg->default_param);
                res.params[arg->short_opt] = arg->default_param;
            } else {
                res.params[arg->short_opt] = *it;
            }
        }
    }
    return res;
}

inline void print_args(const std::vector<Argument> &args, FILE *f = stdout)
{
    const auto maxwidth = std::max_element(args.begin(), args.end(), [](const auto &p, const auto &q) {
        return p.long_opt.size() < q.long_opt.size();
    })->long_opt.size();

    fmt::print(f, "Valid arguments:\n");
    for (const auto &arg : args) {
        fmt::print(f, "    -{}, --{:{}}    {}\n", arg.short_opt, arg.long_opt, maxwidth, arg.desc);
    }
}

inline Result parse(int argc, char **argv, const ArgumentList &valid_args)
{
    return parse(std::vector<std::string_view>{argv, argv + argc}, valid_args);
}

} // namespace cmdline
