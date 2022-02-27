#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <charconv>
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <bitset>
#include <span>
#include <optional>
#include <fmt/core.h>
#include <SDL2/SDL.h>
#include "cmdline.hpp"


/* constants */

const int NUM_PARTICLES = 10;
const int TICK_INTERVAL = 30;
const int SCREEN_WIDTH = 1200;
const int SCREEN_HEIGHT = 1000;

int real_screen_width = SCREEN_WIDTH;
int real_screen_height = SCREEN_HEIGHT;



/* typedefs and vector class */

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using i8 = int8_t;
using i16 = uint16_t;
using i32 = int32_t;
using i64 = int64_t;

template <typename T, unsigned Dim> struct VecData;

#define VECDATA(dim, ...)                               \
    template <typename T> struct VecData<T, dim> {      \
        union {                                         \
            T d[2];                                     \
            struct { __VA_ARGS__ };                     \
        };                                              \
                                                        \
        constexpr VecData(auto... args) : d(args...) {} \
    };

VECDATA(2, T x; T y;)
VECDATA(3, T x; T y; T z;)

#undef VECDATA

template <typename T, unsigned Dim>
struct Vec : VecData<T, Dim> {
    constexpr Vec() = default;
    constexpr Vec(auto... args) : VecData<T, Dim>(args...) {}

    constexpr const T & operator[](std::size_t pos) const { return this->d[pos]; }
    constexpr T & operator[](std::size_t pos) { return this->d[pos]; }
    constexpr T *data()                       { return this->d; }
    constexpr const T *data() const           { return this->d; }
    constexpr std::size_t size() const        { return Dim; }
    constexpr T *begin()                      { return this->d; }
    constexpr T *end()                        { return this->d + Dim; }
    constexpr const T *begin() const          { return this->d; }
    constexpr const T *end() const            { return this->d + Dim; }

    constexpr T length_squared() const
    {
        T res = 0;
        for (auto i = 0u; i < Dim; i++)
            res += this->d[i] * this->d[i];
        return res;
    }

    constexpr T length() const
    {
        if constexpr(std::is_same_v<T, float>) return sqrtf(length_squared());
        return std::sqrt(length_squared());
    }

    Vec<T, Dim> unit() { return *this / length(); }

#define VEC_OP(op) \
    constexpr Vec<T, Dim> & operator op(const Vec<T, Dim> &v)   \
    {                                                           \
        for (auto i = 0u; i < Dim; i++)                         \
            this->d[i] op v.d[i];                               \
        return *this;                                           \
    }
    VEC_OP(+=) VEC_OP(-=) VEC_OP(%=)
#undef VEC_OP
};

#define VEC_OP(op)                                                      \
    template <typename T, unsigned Dim>                                 \
    constexpr Vec<T, Dim> operator op(const Vec<T, Dim> &v, T scalar)   \
    {                                                                   \
        Vec<T, Dim> res;                                                \
        for (auto i = 0u; i < Dim; i++)                                 \
            res[i] = v[i] op scalar;                                    \
        return res;                                                     \
    }
VEC_OP(+) VEC_OP(-) VEC_OP(*) VEC_OP(/)
#undef VEC_OP

#define VEC_OP(op)                                                                  \
    template <typename T, unsigned Dim>                                             \
    constexpr Vec<T, Dim> operator op(const Vec<T, Dim> &v1, const Vec<T, Dim> &v2) \
    {                                                                               \
        Vec<T, Dim> res;                                                            \
        for (auto i = 0u; i < Dim; i++)                                             \
            res[i] = v1[i] op v2[i];                                                \
        return res;                                                                 \
    }
VEC_OP(+) VEC_OP(-)
#undef VEC_OP

template <typename T, unsigned Dim>
T vec_dot(Vec<T, Dim> a, Vec<T, Dim> b)
{
    float res = 0.0f;
    for (auto i = 0u; i < Dim; i++)
        res += a[i] * b[i];
    return res;
}

template <typename T, unsigned Dim>
Vec<T, Dim> vec_lerp(Vec<T, Dim> a, Vec<T, Dim> b, float t)
{
    Vec<T, Dim> res;
    for (auto i = 0u; i < Dim; i++)
        res[i] = std::lerp(a[i], b[i], t);
    return res;
}

template <typename T, unsigned Dim>
auto vec_cross(Vec<T, Dim> a, Vec<T, Dim> b)
{
    if constexpr(Dim == 2) { return a[0]*b[1] - a[1]*b[0]; }
    if constexpr(Dim == 3) { return Vec<T, Dim>{ a[1]*b[2] - a[2]*b[1],
                                                 a[2]*b[0] - a[0]*b[2],
                                                 a[0]*b[1] - a[1]*b[0]  }; }
    return Vec<T, Dim>{};
}

using vec2 = Vec<float, 2>;



/* geometry */

struct Rect {
    vec2 pos;
    vec2 size;

    Rect() = default;
    Rect(vec2 p, vec2 s)                     : pos{p},    size{s}    { }
    Rect(float x, float y, float w, float h) : pos{x, y}, size{w, h} { }

    SDL_Rect to_sdl() { return (SDL_Rect) { int(pos.x), int(pos.y), int(size.x), int(size.y) }; }
};

struct Circle {
    vec2 center;
    float radius;

    Circle() = default;
    Circle(vec2 c, float r)       : center{c},    radius{r} { }
    Circle(float x, float y, float r) : center{x, y}, radius{r} { }

    vec2 top_left() const { return center - radius; }
    vec2 bottom_right() const { return center + radius; }
    vec2 left() const     { return vec2{center.x - radius, center.y}; }
    vec2 right() const    { return vec2{center.x + radius, center.y}; }
    vec2 up() const       { return vec2{center.x, center.y - radius}; }
    vec2 down() const     { return vec2{center.x, center.y + radius}; }
};

bool circle_point_intersecting(Circle c, vec2 p)
{
    return (c.center - p).length_squared() <= c.radius;
}

bool circle_circle_intersecting(Circle c1, Circle c2)
{
    return (c1.center - c2.center).length_squared() <= powf(c1.radius + c2.radius, 2);
}

bool rect_point_intersecting(Rect r, vec2 p)
{
    return p.x > r.pos.x && p.x < r.pos.x + r.size.x
        && p.y > r.pos.y && p.y < r.pos.y + r.size.y;
}

bool rect_rect_intersecting(Rect r1, Rect r2)
{
    return r1.pos.x < r2.pos.x + r2.size.x
        && r2.pos.x < r1.pos.x + r1.size.x
        && r1.pos.y < r2.pos.y + r2.size.y
        && r2.pos.y < r1.pos.y + r1.size.y;
}



/* rng */

// (m, n]
float random_between(float m, float n)
{
    assert(n - m > 0);
    return float((rand() % int(n - m)) + m);
}

vec2 random_inside(Rect r)
{
    return {
        random_between(r.pos.x, r.pos.x + r.size.x),
        random_between(r.pos.y, r.pos.y + r.size.y),
    };
}

float random_no_zero(float n)
{
    float nums[2];
    nums[0] = random_between(-n, 0);
    nums[1] = random_between(1, n+1);
    int which = int(random_between(0, 2));
    return nums[which];
}



/* collision */

std::optional<float> ccd_circle_line(float p0, float p1, float r, float b)
{
    float tc1 = (b + r - p0) / (p1 - p0);
    float tc2 = (b - r - p0) / (p1 - p0);
    float tc = std::min(tc1, tc2);
    if (tc < 0.0f || tc > 1.0f)
        return std::nullopt;
    float pc = std::lerp(p0, p1, tc);
    return pc*2 - p1;
}

float pow2f(float n) { return n*n; }

std::pair<vec2, vec2> elastic_collision_2d(vec2 v1, vec2 v2, float m1, float m2, vec2 c1, vec2 c2)
{
    auto f = [](vec2 v1, vec2 v2, float m1, float m2, vec2 c1, vec2 c2) {
        return v1 - (c1 - c2) * (2.0f * m2 / (m1 + m2)) * vec_dot(v1 - v2, c1 - c2) / pow2f((c1 - c2).length());
    };
    vec2 v1p = f(v1, v2, m1, m2, c1, c2);
    vec2 v2p = f(v2, v1, m2, m1, c2, c1);
    return std::make_pair(v1p, v2p);
}



/* particle and engine classes */

struct Graphics {
    SDL_Texture *tex;
    vec2 size;
};

struct {
    std::vector<Graphics> loaded_gfx;

    int add(Graphics &&gfx)
    {
        loaded_gfx.emplace_back(gfx);
        return loaded_gfx.size() - 1;
    }

    Graphics & operator[](int id) { return loaded_gfx[id]; }
    vec2 gfx_size(int id)         { return loaded_gfx[id].size; }
} gfx_handler;

struct Particle {
    Circle hitbox;
    vec2 vel = {0,0};
    int gfx_id; // which Graphics to use
    float mass = 1.0f;

    Particle(vec2 p, float r, vec2 v, int id)
        : hitbox{p, r}, vel{v}, gfx_id{id}
    { }

    void update();
    vec2 adjust_pos(vec2 p0, vec2 p1, float r);
    const vec2 & pos() const { return hitbox.center; }
};

std::pair<vec2, vec2> particle_collision(const Particle &p, const Particle &q)
{
    return elastic_collision_2d(p.vel, q.vel, p.mass, q.mass, p.hitbox.center, q.hitbox.center);
}

struct Engine {
    SDL_Window *window;
    SDL_Renderer *rd;
    std::vector<Particle> particles;
    int wnd_width = SCREEN_WIDTH, wnd_height = SCREEN_HEIGHT;

    void init(int width, int height);
    void deinit();
    bool poll();
    void draw_one(vec2 pos, int gfx_id);
    void draw();
    int load_gfx(std::string_view pathname);
    void tick();
    void add_sprite(Particle particle);
    void draw_uniform_grid_lines(int grid_size);

    int screen_width() const  { return wnd_width; }
    int screen_height() const { return wnd_height; }
} engine;



/* collisions (broad phase) */

using Collision = std::pair<Particle *, Particle *>;

std::vector<Collision> broad_phase_brute(std::span<Particle> particles)
{
    std::vector<Collision> collisions;
    for (auto i = 0u; i < particles.size()-1; i++)
        for (auto j = i+1; j < particles.size(); j++)
            if (circle_circle_intersecting(particles[i].hitbox, particles[j].hitbox))
                collisions.push_back(std::make_pair(&particles[i], &particles[j]));
    return collisions;
}

template <typename T>
void filter(T &v, auto &&fn)
{
    auto it = std::remove_if(v.begin(), v.end(), fn);
    v.erase(it, v.end());
}

std::vector<Collision> sweep_and_prune(std::span<Particle> particles)
{
    std::vector<Particle *> axis_list{particles.size()};
    for (auto i = 0u; i < particles.size(); i++)
        axis_list[i] = &particles[i];
    std::sort(axis_list.begin(), axis_list.end(), [](const auto &p, const auto &q) {
        return p->hitbox.center.x < q->hitbox.center.x;
    });

    std::vector<Particle *> active;
    std::vector<Collision> collisions;
    for (auto *p : axis_list) {
        filter(active, [&](const auto *q) {
            return p->hitbox.left().x > q->hitbox.right().x;
        });
        for (auto *q : active)
            collisions.push_back(std::make_pair(p, q));
        active.push_back(p);
    }

    return collisions;
}

void test_sap()
{
    std::vector<Particle> ps = {
        Particle{ vec2{10.0f, 40.0f},  5.0f, vec2{}, 0 },
        Particle{ vec2{15.0f, 100.0f}, 5.0f, vec2{}, 0 },
        Particle{ vec2{40.0f, 15.0f},  5.0f, vec2{}, 0 },
        Particle{ vec2{60.0f, 40.0f},  5.0f, vec2{}, 0 },
        Particle{ vec2{65.0f, 35.0f},  5.0f, vec2{}, 0 },
        Particle{ vec2{76.0f, 90.0f},  5.0f, vec2{}, 0 },
    };
    auto collisions = sweep_and_prune(ps);
    for (auto &c : collisions) {
        auto &p = c.first;
        auto &q = c.second;
        fmt::print("collision between {} ({}, {}) and {} ({}, {})\n",
            uintptr_t(p), p->hitbox.center.x, p->hitbox.center.y,
            uintptr_t(q), q->hitbox.center.x, q->hitbox.center.y);
    }
}

template <typename T, u32 N>
struct Grid {
    struct PairHash {
        std::size_t operator()(const std::pair<u32, u32> &p) const
        {
            return p.first * N + p.second;
        }
    };

    std::unordered_map<std::pair<u32, u32>,
                       std::vector<T>,
                       PairHash> data;

    void put_in_cell(u32 x, u32 y, T t)
    {
        // fmt::print("adding at {},{}\n", x, y);
        assert(x < N && y < N);
        auto &v = data[{x, y}];
        v.push_back(t);
    }

    void put(vec2 tl, vec2 br, T t)
    {
        int start_row = std::max(0, (int) tl.x / (engine.screen_width() / 3));
        int start_col = std::max(0, (int) tl.y / (engine.screen_width() / 3));
        int end_row   = std::min(2, (int) br.x / (engine.screen_height() / 3));
        int end_col   = std::min(2, (int) br.y / (engine.screen_height() / 3));
        for (int row = start_row; row <= end_row; row++)
            for (int col = start_col; col <= end_col; col++)
                put_in_cell(row, col, t);
    }

    auto begin()       { return data.begin(); }
    auto end()         { return data.end(); }
};

std::vector<Collision> uniform_grid(std::span<Particle> particles)
{
    Grid<Particle *, 8> grid;
    for (auto &p : particles)
        grid.put(p.hitbox.top_left(), p.hitbox.bottom_right(), &p);

    std::vector<Collision> collisions;
    for (auto [pos, cell] : grid)
        for (auto i = 0u; i < cell.size()-1; i++)
            for (auto j = i+1; j < cell.size(); j++)
                collisions.push_back(std::make_pair(cell[i], cell[j]));

    return collisions;
}

std::vector<Collision> broad_phase(std::span<Particle> particles)
{
    auto possible = sweep_and_prune(particles);
    filter(possible, [&](const auto &c) {
        auto &p = *c.first;
        auto &q = *c.second;
        return !circle_circle_intersecting(p.hitbox, q.hitbox);
    });
    return possible;
}

/*
 * note that the k-d tree algorithm, as implemented like this, won't take count
 * of the fact that some particles might be positioned right where the split
 * occurs, making these particles part of both splitted areas.
 * (this means it doesn't do shit and thus should only be taken as a
 * demonstration on how to start implementing it. i've spent enough time with it
 * already, i don't feel like finishing it).
 */
void kdtree(std::span<Particle *> particles, int depth, auto &&fn)
{
    if (particles.size() <= 1)
        return;
    if (depth > 10) {
        fn(particles);
        return;
    }
    auto axis = depth % 2;
    auto median = particles.size() / 2;
    std::sort(particles.begin(), particles.end(), [&](const auto *p, const auto *q) {
        return p->pos()[axis] < q->pos()[axis];
    });
    kdtree(particles.subspan(0, median), depth + 1, fn);
    kdtree(particles.subspan(median),    depth + 1, fn);
}

std::vector<Collision> broad_phase_kdtree(std::span<Particle> particles)
{
    std::vector<Particle *> pointers;
    std::vector<Collision> collisions;
    for (auto &p : particles)
        pointers.push_back(&p);
    kdtree(pointers, 0, [&](std::span<Particle *> ps) {
        for (auto i = 0u; i < ps.size(); i++) {
            for (auto j = i+1; j < ps.size(); j++) {
                auto &p = ps[i];
                auto &q = ps[j];
                if (circle_circle_intersecting(p->hitbox, q->hitbox)) {
                    collisions.push_back(std::make_pair(p, q));
                }
            }
        }
    });
    return collisions;
}



/*
 * particle and engine implementations
 * (put here because one particle function depends on engine class
 *  and engine class depends on particle class)
 */

void Particle::update()
{
    vec2 p1 = hitbox.center;
    vec2 p2 = hitbox.center + vel;
    vec2 new_center = adjust_pos(p1, p2, hitbox.radius);
    hitbox.center = new_center;
}

vec2 Particle::adjust_pos(vec2 p0, vec2 p1, float r)
{
    vec2 res = p1;
    if (auto wall = ccd_circle_line(p0.x, p1.x, r, 0);                      wall) { res.x = wall.value(); vel.x = -vel.x; }
    if (auto wall = ccd_circle_line(p0.x, p1.x, r, engine.screen_width());  wall) { res.x = wall.value(); vel.x = -vel.x; }
    if (auto wall = ccd_circle_line(p0.y, p1.y, r, 0);                      wall) { res.y = wall.value(); vel.y = -vel.y; }
    if (auto wall = ccd_circle_line(p0.y, p1.y, r, engine.screen_height()); wall) { res.y = wall.value(); vel.y = -vel.y; }
    return res;
}

void Engine::init(int width, int height)
{
    SDL_Init(SDL_INIT_VIDEO);
    window = SDL_CreateWindow("Particle sim", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, width, height, SDL_WINDOW_SHOWN);
    rd = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    wnd_width = width;
    wnd_height = height;
}

void Engine::deinit()
{
    SDL_DestroyRenderer(rd);
    SDL_DestroyWindow(window);
}

bool Engine::poll()
{
    for (SDL_Event ev; SDL_PollEvent(&ev); ) {
        switch (ev.type) {
        case SDL_QUIT:
            return false;
            break;
        }
    }
    return true;
}

void Engine::draw_one(vec2 pos, int gfx_id)
{
    auto &gfx = gfx_handler[gfx_id];
    vec2 size = gfx.size;
    SDL_Rect src = { 0, 0, int(size.x), int(size.y) };
    SDL_Rect dst = { int(pos.x), int(pos.y), int(size.x), int(size.y) };
    SDL_RenderCopy(rd, gfx.tex, &src, &dst);
}

void Engine::draw()
{
    SDL_SetRenderDrawColor(rd, 0, 0, 0, 0xff);
    SDL_RenderClear(rd);
    for (auto &p : particles)
        draw_one(p.hitbox.top_left(), p.gfx_id);
    SDL_SetRenderDrawColor(rd, 0xff, 0, 0, 0xff);
    draw_uniform_grid_lines(8);
    SDL_RenderPresent(rd);
}

void Engine::draw_uniform_grid_lines(int grid_size)
{
    for (int i = 1; i < grid_size; i++) {
        vec2 start = { screen_width() / grid_size * i, 0 };
        vec2 end   = { screen_width() / grid_size * i, screen_height() };
        SDL_RenderDrawLine(rd, int(start.x), int(start.y), int(end.x), int(end.y));
    }

    for (int i = 1; i < grid_size; i++) {
        vec2 start = { 0,              screen_height() / grid_size * i };
        vec2 end   = { screen_width(), screen_height() / grid_size * i };
        SDL_RenderDrawLine(rd, int(start.x), int(start.y), int(end.x), int(end.y));
    }
}

int Engine::load_gfx(std::string_view pathname)
{
    auto *bmp = SDL_LoadBMP(pathname.data());
    assert(bmp && "load of bmp image failed");
    auto *tex = SDL_CreateTextureFromSurface(rd, bmp);
    SDL_FreeSurface(bmp);
    return gfx_handler.add({ .tex = tex, .size = {bmp->w, bmp->h} });
}

void Engine::tick()
{
    for (auto &particle : particles)
        particle.update();
    auto collisions = broad_phase(particles);
    for (auto [p, q] : collisions) {
        auto [v1, v2] = particle_collision(*p, *q);
        p->vel = v1;
        q->vel = v2;
    }
}

void Engine::add_sprite(Particle particle)
{
    particles.push_back(particle);
}



/* clock */

struct Clock {
    u32 next = 0;

    void init() { next = SDL_GetTicks() + TICK_INTERVAL; }
    void update() { next += TICK_INTERVAL; }

    u32 time_left()
    {
        u32 now = SDL_GetTicks();
        return next <= now ? 0 : next - now;
    }
} game_clock;



/* main stuff */

auto generate_particle(int screen_width, int screen_height, int id)
{
    float radius = float(gfx_handler.gfx_size(id).x / 2);
    vec2 pos{
        random_between(0 + radius, screen_width  - radius),
        random_between(0 + radius, screen_height - radius),
    };
    vec2 vel{ random_no_zero(6), random_no_zero(6) };
    return Particle{pos, radius, vel, id};
}

void init(int width, int height, int num_particles)
{
    std::srand(time(nullptr));
    engine.init(width, height);
    std::array<int, 5> gfx;
    gfx[0]  = engine.load_gfx("orange.bmp");
    gfx[1]  = engine.load_gfx("blue.bmp");
    gfx[2]  = engine.load_gfx("cyan.bmp");
    gfx[3]  = engine.load_gfx("red.bmp");
    gfx[4]  = engine.load_gfx("green.bmp");

    // float radius = float(gfx_handler.gfx_size(0).x / 2);
    // engine.add_sprite(Particle{ vec2{width / 8, 20}, radius, vec2{0, 5}, 0});
    // engine.add_sprite(Particle{ vec2{width / 8, 400}, radius, vec2{0, -5}, 0});

    for (int i = 0; i < num_particles; i++) {
        int id = gfx[random_between(0, 5)];
        engine.add_sprite(generate_particle(width, height, id));
    }

    game_clock.init();
}

void deinit()
{
    engine.deinit();
}

void game_loop()
{
    for (bool running = true; running; ) {
        running = engine.poll();
        engine.tick();
        engine.draw();
        SDL_Delay(game_clock.time_left());
        game_clock.update();
    }
}

template <typename T>
std::optional<int> to_int(const T &str, unsigned base = 10)
{
    int value = 0;
    auto res = std::from_chars(str.data(), str.data() + str.size(), value, base);
    if (res.ec != std::errc() || res.ptr != str.data() + str.size())
        return std::nullopt;
    return value;
}

int main(int argc, char *argv[])
{
    static const cmdline::ArgumentList args = {
        { 'h', "help", "print this help text" },
        { 'n', "num-particles", "set number of particles", cmdline::ParamType::Single },
        { 'w', "width", "set screen width", cmdline::ParamType::Single },
        { 'i', "height", "set screen height", cmdline::ParamType::Single },
    };
    auto result = cmdline::parse(argc, argv, args);
    if (result.has['h']) {
        cmdline::print_args(args);
        return 0;
    }
    int width = SCREEN_WIDTH, height = SCREEN_HEIGHT, num = NUM_PARTICLES;
    if (result.has['w']) {
        auto o = to_int(result.params['w']);
        if (!o) {
            fmt::print(stderr, "invalid number: {}\n", result.params['w']);
            return 1;
        }
        width = o.value();
    }
    if (result.has['i']) {
        auto o = to_int(result.params['i']);
        if (!o) {
            fmt::print(stderr, "invalid number: {}\n", result.params['i']);
            return 1;
        }
        height = o.value();
    }
    if (result.has['n']) {
        auto o = to_int(result.params['n']);
        if (!o) {
            fmt::print(stderr, "invalid number: {}\n", result.params['n']);
            return 1;
        }
        num = o.value();
    }

    init(width, height, num);
    game_loop();
    deinit();
    return 0;
}
