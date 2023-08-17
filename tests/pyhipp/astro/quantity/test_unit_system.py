from pyhipp.astro.quantity import UnitSystem as US

def test_init():
    us = US.create_for_cosmology(.7)
    
    u = us.astropy_u
    assert isinstance((us.u_v / (u.km / u.s)).to(1).value, float)