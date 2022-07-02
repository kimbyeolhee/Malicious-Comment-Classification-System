import styled from "styled-components";

const Container = styled.div`
  box-shadow: 0 0 0 1px rgb(87 87 87 / 10%), 0 8px 8px 0 rgb(234 224 218 / 30%);
  width: 800px;
  border-radius: 15px;
  box-sizing: border-box;
  background-color: #fff;
`;

function PostList() {
  const dummy = [
    { title: "제목1", count: 3 },
    { title: "제목2", count: 4 },
    { title: "제목3", count: 5 },
  ];
  return (
    <>
      <Container>
        <ul class="list-group">
          {dummy.map((post) => (
            <li class="list-group-item d-flex justify-content-between align-items-center">
              {post.title}
              <span class="badge bg-primary rounded-pill">{post.count}</span>
            </li>
          ))}
        </ul>
      </Container>
    </>
  );
}

export default PostList;
