import React from "react";
import Navbar from "../components/Navbar";
import Hero from "../components/Hero";
import News from "../components/News";
import Doctor from "../components/Doctor";
import Appointment from "../components/Appointment";
import Services from "../components/Services";
import Footer from "../components/Footer";
import { useState,useEffect } from "react";
const Home = () => {
  const [token, settoken] = useState("");
  useEffect(() => {
    const user = JSON.parse(localStorage.getItem("user"));
    if (user && user.jsontoken) {
      settoken(user.jsontoken);
    }
  }, [token]);
  return (
    <>
      <Navbar />
      <Hero title="Healthy Living" para="Let's make your life happier" />
      <Services/>
      <Doctor />
      <News />
     
      {token!=""?  <Appointment />:""}
      <Footer/>
    </>
  );
};
export default Home;
