package com.example.mybatisplusdemo.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.example.mybatisplusdemo.model.domain.User;
import com.example.mybatisplusdemo.mapper.UserMapper;
import com.example.mybatisplusdemo.model.dto.UserQueryDTO;
import com.example.mybatisplusdemo.service.IUserService;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

/**
 * <p>
 *  服务实现类
 * </p>
 *
 * @author lxp
 * @since 2025-06-17
 */
@Service
public class UserServiceImpl extends ServiceImpl<UserMapper, User> implements IUserService {

    @Autowired
    private UserMapper userMapper;

    @Override
    public User getByIdMy(Long id) {
        User user = userMapper.selectById(id);
        return user;
    }

    @Override
    public Page<User> page(Page<User> page, QueryWrapper<User> wrapper) {
        return baseMapper.selectPage(page, wrapper);
    }

    @Override
    public Page<User> selectPage(UserQueryDTO userQueryDTO) {
        Page<User> page=new Page<>(userQueryDTO.getPageNo(),userQueryDTO.getPageSize());
        return userMapper.selectPageOwn(page, userQueryDTO);
    }

    @Override
    public List<User> listByKey(String key) {
        LambdaQueryWrapper<User> wrapper=new LambdaQueryWrapper<>();
        wrapper.like(User::getLoginName,key);
        return userMapper.selectList(wrapper);
    }

    //测试

}
