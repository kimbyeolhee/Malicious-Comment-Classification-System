import React, { useContext, useState, useEffect } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { UserStateContext } from "../App";
import * as Api from "../api";

import PostList from "./community/PostList";
import styled from "styled-components";
import { Link } from "react-router-dom";

const MainContainer = styled.div`
  background-color: #f8f9fa;
`;

const Container = styled.div`
  display: flex;
  position: relative;
  width: 1200px;
  padding: 120px 48px;
  margin: 0 auto;
`;

const BoxWrapper = styled.div`
  display: flex;
  width: 90%;
  justify-content: center;
  align-items: center;
`;

const StepWrapper = styled.div`
  display: flex;
  margin-right: 100px;
  height: 100px;
  width: 100px;
`;

function PostMain() {
  const navigate = useNavigate();
  const params = useParams();

  const [isFetchCompleted, setIsFetchCompleted] = useState(false);
  const userState = useContext(UserStateContext);

  const fetchOwner = async (ownerId) => {
    const res = await Api.get("users", ownerId);
    const ownerData = res.data;
    setIsFetchCompleted(true);
  };

  useEffect(() => {
    if (!userState.user) {
      navigate("/login", { replace: true });
      return;
    }

    if (params.userId) {
      const ownerId = params.userId;
      fetchOwner(ownerId);
    } else {
      const ownerId = userState.user.id;
      fetchOwner(ownerId);
    }
  }, [params, userState, navigate]);

  if (!isFetchCompleted) {
    return "loading...";
  }

  return (
    <>
      <title>커뮤니티</title>

      <MainContainer>
        <Container>
          <BoxWrapper>
            <PostList />
          </BoxWrapper>
          <StepWrapper>
            <Link to={`/community/newPost`}>
              <button type="button" class="btn btn-primary btn-sm">
                게시글 작성
              </button>
            </Link>
          </StepWrapper>
        </Container>
      </MainContainer>
    </>
  );
}

export default PostMain;
