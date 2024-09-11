const express = require("express");
const routes = express.Router();
const user=require("../models/User");
routes.post("/authorize",async(req,res)=>{
  console.log(req.body.token)
  try{
  const getuser= await user.findOne({JwtToken:req.body.token});
  if(getuser){
    return res.send({success:true,user:getuser});
  }else{
    return res.send({success:false, message:"error"})
  }
  }catch (e){
    return res.send({success:false, message:"Catch error"})
  }
})
module.exports=routes