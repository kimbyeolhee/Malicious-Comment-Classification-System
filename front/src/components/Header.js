import React, { useContext } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import Nav from "react-bootstrap/Nav";
import { UserStateContext, DispatchContext } from "../App";

function Header() {
  const navigate = useNavigate();
  const location = useLocation();

  const userState = useContext(UserStateContext);
  const dispatch = useContext(DispatchContext);

  const isLogin = !!userState.user;

  const logout = () => {
    sessionStorage.removeItem("userToken");
    dispatch({ type: "LOGOUT" });
    navigate("/");
  };

  return (
    <Nav activeKey={location.pathname}>
      <Nav.Item className="me-auto mb-5">
        <Nav.Link disabled>악성댓글 분류 프로젝트</Nav.Link>
      </Nav.Item>
      <Nav.Item>
        <Nav.Link onClick={() => navigate("/")}>나의 페이지</Nav.Link>
      </Nav.Item>
      {isLogin && (
        <Nav.Item>
          <Nav.Link onClick={logout}>로그아웃</Nav.Link>
        </Nav.Item>
      )}
    </Nav>
  );
}

export default Header;
