import React, { useEffect, useState } from "react";
import {
  FaFacebookSquare,
  FaInstagramSquare,
  FaTwitter,
  FaPhoneAlt,
} from "react-icons/fa";
import { RiAccountCircleLine } from "react-icons/ri";
import { IoMail } from "react-icons/io5";
import { CiSearch } from "react-icons/ci";
import { Link, useNavigate } from "react-router-dom";
const Navbar = () => {
  const navigate = useNavigate();
  const [token, settoken] = useState("");
  useEffect(() => {
    const user = JSON.parse(localStorage.getItem("user"));
    if (user && user.jsontoken) {
      settoken(user.jsontoken);
    }
  }, [token]);

  const handlelogout = () => {
    localStorage.removeItem("user");
    settoken("");
    navigate("/");
  };
  return (
    <>
      <header>
        <div className="topbar">
          <div className="container">
            <div className="row">
              <div className="col-lg-6 col-md-12 text-sm">
                <div className="site-info">
                  <span className="mai-call text-secondary">
                    <FaPhoneAlt />
                    +00 123 4455 6666
                  </span>

                  <span className="divider">|</span>
                  <span className="mai-mail text-secondary">
                    <IoMail />
                    mail@example.com
                  </span>
                </div>
              </div>
              <div className="col-lg-6 col-md-12 text-sm text-secondary">
                <div className="social-mini-button fs-5 d-flex justify-content-evenly">
                  <Link to="/" aria-label="Facebook">
                    <FaFacebookSquare className="text-secondary" />
                  </Link>
                  <Link to="/" aria-label="Instagram">
                    <FaInstagramSquare className="text-secondary" />
                  </Link>
                  <Link to="/" aria-label="Twitter">
                    <FaTwitter className="text-secondary" />
                  </Link>
                </div>
              </div>
            </div>
          </div>
        </div>
        {/* <hr /> */}
        <nav className="navbar navbar-expand-lg navbar-light shadow-sm p-0">
          <div className="container d-flex justify-content-between">
            {/* Brand */}
            <a className="navbar-brand fs-3" href="#">
              <span className="text-bold text-primary fs-2">One</span>-Health
            </a>

            {/* Search Form */}
            <form action="#" className="d-flex flex-grow-1 mx-5">
              <div
                className="input-group input-navbar"
                style={{ border: "1px solid black" }}
              >
                <div className="input-group-prepend">
                  <span
                    className="bg-none fs-4 text-primary p-2"
                    id="icon-addon1"
                  >
                    <span className="mai-search">
                      <CiSearch />
                    </span>
                  </span>
                </div>
                <input
                  type="text"
                  className="form-control fs-5 p-0"
                  placeholder="Enter keyword.."
                  aria-label="Username"
                  aria-describedby="icon-addon1"
                  style={{ borderLeft: "1px solid black" }}
                />
              </div>
            </form>
            {/* Toggler Button */}
            <button
              className="navbar-toggler"
              type="button"
              data-bs-toggle="collapse"
              data-bs-target="#navbarSupport"
              aria-controls="navbarSupport"
              aria-expanded="false"
              aria-label="Toggle navigation"
            >
              <span className="navbar-toggler-icon"></span>
            </button>
            {/* Navigation Links */}
            <div className="collapse navbar-collapse" id="navbarSupport">
              <ul className="navbar-nav ms-auto">
                <li className="nav-item active">
                  <Link to="/" className="nav-link">
                    Home
                  </Link>
                </li>
                <li className="nav-item">
                  <Link to="/about" className="nav-link">
                    AboutUs
                  </Link>
                </li>
                <li className="nav-item">
                  {token != "" ? (
                    <>
                      <ul className="navbar-nav ms-auto">
                        <li className="nav-item">
                          <Link to="/news" className="nav-link">
                            News
                          </Link>
                        </li>
                        <li className="nav-item">
                          <Link to="/contact" className="nav-link">
                            Contact
                          </Link>
                        </li>
                        <li>
                          <div class="dropdown">
                            <button
                              style={{ border: "none", background: "none" }}
                              type="button"
                              className="fs-4"
                              data-bs-toggle="dropdown"
                              aria-expanded="false"
                            >
                              <RiAccountCircleLine />
                            </button>
                            <ul class="dropdown-menu">
                              <li>
                                <Link
                                  style={{
                                    textdecoration: "none",
                                    color: "black",
                                  }}
                                  onClick={handlelogout}
                                >
                                  Log out
                                </Link>
                              </li>
                            </ul>
                          </div>
                          {/*  */}
                        </li>
                      </ul>
                    </>
                  ) : (
                    <Link className="btn btn-primary ml-lg-3" to="/login">
                      Login / Register
                    </Link>
                  )}
                </li>
              </ul>
            </div>
          </div>
        </nav>
      </header>
    </>
  );
};
export default Navbar;
