using MeshCat
using CoordinateTransformations
using GeometryBasics
using Colors

function set_mesh!(vis, ::YakPlane; color=nothing, scale=0.15)
    meshfile = joinpath(@__DIR__,"..","data","piper","piper_pa18.obj")
    # meshfile = joinpath(@__DIR__,"..","data","meshes","cirrus","Cirrus.obj")
    # meshfile = joinpath(@__DIR__,"..","data","meshes","piper","piper_scaled.obj")
    jpg = joinpath(@__DIR__,"..","data","piper","piper_diffuse.jpg")
    if isnothing(color)
        img = PngImage(jpg)
        texture = Texture(image=img)
        # mat = MeshLambertMaterial(map=texture) 
        mat = MeshPhongMaterial(map=texture) 
    else
        mat = MeshPhongMaterial(color=color)
    end
    obj = MeshFileGeometry(meshfile)
    setobject!(vis["robot"]["geom"], obj, mat)
    settransform!(vis["robot"]["geom"], compose(Translation(0,0,0.07),LinearMap( RotY(pi/2)*RotZ(-pi/2) * scale)))
end

function dynamics(model::YakPlane, x, u)
    RobotDynamics.dynamics(model, SVector{13}(x), SVector{4}(u))
end

function discrete_dynamics(model::YakPlane, x, u, t, dt)
    RobotDynamics.discrete_dynamics(RobotDynamics.RK4, model, SVector{13}(x), SVector{4}(u), t, dt)
end

function discrete_jacobian(model::YakPlane, x, u, t, dt)
    ∇f = RobotDynamics.DynamicsJacobian(model)
    z = RobotDynamics.StaticKnotPoint(SVector{13}(x), SVector{4}(u), dt, t)
    RobotDynamics.discrete_jacobian!(RobotDynamics.RK4, ∇f, model, z)
    return RobotDynamics.get_static_A(∇f), RobotDynamics.get_static_B(∇f)
end